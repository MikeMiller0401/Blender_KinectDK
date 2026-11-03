"""Azure Kinect DK 模拟：渲染虚拟 DK 的彩色相机视角与深度相机视角（含红外照明），供 kinect_sim.py 加噪。
用法：blender -b apartment.blend --python render_kinect.py -- 输出目录 [--views 名称=x,y,z,lx,ly,lz;... | mirrors]
        [--samples 128] [--color-samples 128]
内外参取自真机出厂标定（factory_calibration.json，NFOV unbinned 640×576 + 720p 彩色）与 dk_params.json。
每个视角输出 <名称>/：
  color.png         彩色相机 1280×720（针孔，不含镜头畸变）
  ir_passes.npz     深度相机针孔视角（覆盖畸变后的视场）：只开相机处红外点光源时的
                    Combined 与漫反射/高光/透射的颜色、直接光、间接光通道，Z 深度、世界法线、物体 ID
  through.npz       隐去玻璃后的 Z 深度（ToF 透过玻璃测到后方）
  mirror.npz        经镜面反射的像素：沿视线展开后的总路程（虚像距离）、首次入射余弦、末端表面余弦、反射次数
另写 rig.json（内外参、针孔渲染参数、玻璃/金属物体 ID）与 labels.json。不保存源文件。
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dk_common import undistort  # noqa: E402

# 厨房测试视角：彩色相机位置与注视点（手持高度）
VIEWS = {
    'kitchen_overview': (9.0, 4.6, 1.45, 11.4, 1.2, 0.85),
    'kitchen_south': (10.3, 3.4, 1.35, 10.6, 0.3, 1.0),
    'kitchen_east': (10.9, 3.0, 1.30, 12.4, 1.6, 0.95),
    'kitchen_glassdoor': (11.3, 2.9, 1.45, 8.4, 3.2, 1.1),
}
# 镜子测试视角（--views mirrors）：正对（中心有发射器镜面亮斑）与斜看
MIRROR_VIEWS = {
    'bed2_mirror_frontal': (11.25, 7.98, 1.40, 12.46, 7.98, 1.15),
    'hall_mirror_oblique': (7.25, 5.18, 1.40, 6.45, 6.33, 1.10),
    'living_mirror_oblique': (7.17, 2.35, 1.40, 8.31, 1.55, 1.24),
    # 镜中物体近（展开路程中位数约 1.5 / 2.0 m），对应真机 mirror04/05
    'bed1_wardrobe_close': (7.11, 7.37, 1.30, 7.72, 6.85, 1.15),
    'bed1_floormirror_close': (5.44, 7.89, 1.20, 4.71, 7.55, 0.80),
}
CV2BL = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))   # OpenCV 相机系 ↔ Blender 相机系


def look_at(loc, target):
    """Blender 相机矩阵：-Z 指向目标，Y 朝上。"""
    loc, target = Vector(loc), Vector(target)
    rot = (target - loc).to_track_quat('-Z', 'Y').to_matrix().to_4x4()
    rot.translation = loc
    return rot


def set_intrinsics(cam, K, w, h):
    """针孔内参写入 Blender 相机（水平适配，主点用 shift 表示）。
    K 为 OpenCV 约定（像素中心在整数坐标），Blender 图像中心对应 (w-1)/2，故主点加 0.5。"""
    cam.type = 'PERSP'
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 36.0
    cam.lens = K[0][0] * 36.0 / w
    cam.shift_x = -(K[0][2] + 0.5 - w / 2) / w
    cam.shift_y = (K[1][2] + 0.5 - h / 2) / w
    cam.clip_start = 0.05


def read_exr(path):
    """读多层 EXR，按通道全名返回 float32 数组（行序与图像一致）。"""
    img = oiio.ImageBuf(str(path))
    spec = img.spec()
    px = img.get_pixels(oiio.FLOAT).reshape(spec.height, spec.width, spec.nchannels)
    return {n: px[..., i] for i, n in enumerate(spec.channelnames)}


def passes(chans, name, comps):
    keys = [k for k in chans if k.split('.')[-2:-1] == [name]]
    return np.stack([chans[next(k for k in keys if k.endswith('.' + c))] for c in comps], -1)


def surface_kind(m):
    """材质类别：'mirror' 近乎理想镜面的金属，'glass' 透射，None 其他。"""
    if not (m and m.use_nodes):
        return None
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            if n.inputs['Transmission Weight'].default_value > 0.5:
                return 'glass'
            if n.inputs['Metallic'].default_value > 0.9 and n.inputs['Roughness'].default_value < 0.05:
                return 'mirror'
        elif n.type == 'BSDF_GLASS':
            return 'glass'
    return None


class MirrorTracer:
    """沿视线追踪镜面反射（scene.ray_cast）：穿过玻璃面，遇镜面按法线反射，直到漫反射表面。
    ToF 的发射与接收都经镜面往返，测得的是展开后的总路程，即镜中虚像的距离。"""

    def __init__(self, scene):
        self.scene = scene
        self.dg = bpy.context.evaluated_depsgraph_get()
        self.faces = {}

    def kind(self, ob, face):
        ob = getattr(ob, 'original', ob)
        if ob.hide_render or not ob.visible_camera:
            return 'skip'
        if ob.name not in self.faces:
            ev = ob.evaluated_get(self.dg)              # 曲线、文字等也按求值后的网格取面
            me = ev.to_mesh()
            mi = np.zeros(len(me.polygons), np.int32)
            me.polygons.foreach_get('material_index', mi)
            ev.to_mesh_clear()
            kinds = [surface_kind(s.material) for s in ob.material_slots] or [None]
            self.faces[ob.name] = (mi, kinds)
        mi, kinds = self.faces[ob.name]
        return kinds[min(mi[face], len(kinds) - 1)] if face < len(mi) else None

    def trace(self, origin, d, max_steps=12):
        """返回 (总路程, 首次镜面入射余弦, 末端表面余弦, 反射次数)；未遇镜面返回 None，反射后射空路程为 inf。"""
        total, bounces, cos0 = 0.0, 0, 0.0
        o = origin
        for _ in range(max_steps):
            hit, loc, n, face, ob, _ = self.scene.ray_cast(self.dg, o, d)
            if not hit:
                return (np.inf, cos0, 0.0, bounces) if bounces else None
            total += (loc - o).length
            k = self.kind(ob, face)
            if k == 'mirror' and bounces < 4:
                c = d.dot(n)
                if bounces == 0:
                    cos0 = abs(c)
                d = (d - 2 * c * n).normalized()
                bounces += 1
            elif k not in ('glass', 'skip'):
                return (total, cos0, abs(d.dot(n)), bounces) if bounces else None
            o = loc + d * 1e-4
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--views')
    ap.add_argument('--samples', type=int, default=128)
    ap.add_argument('--color-samples', type=int, default=128)
    opt = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    views = VIEWS
    if opt.views == 'mirrors':
        views = MIRROR_VIEWS
    elif opt.views:
        views = {k: tuple(float(x) for x in v.split(',')) for k, v in (s.split('=') for s in opt.views.split(';'))}

    calib = json.loads((HERE / 'factory_calibration.json').read_text())['CalibrationInformation']
    pv = next(c for c in calib['Cameras'] if c['Purpose'] == 'CALIBRATION_CameraPurposePhotoVideo')
    R = np.array(pv['Rt']['Rotation']).reshape(3, 3)
    t = np.array(pv['Rt']['Translation'])
    T_c_d = np.eye(4)                       # 深度相机系 → 彩色相机系（OpenCV，米）
    T_c_d[:3, :3], T_c_d[:3, 3] = R, t
    params = json.loads((HERE / 'dk_params.json').read_text())
    K_d = np.array(params['K_depth'])
    dist = np.array(params['dist_depth_k1k2p1p2k3k4k5k6'])
    K_c = np.loadtxt('/home/tcy/Programs/AzureKinectDK_Ubuntu26/output/dry_pyk4a_export/intrinsic_color.txt')
    Wd, Hd, Wc, Hc = 640, 576, 1280, 720

    # 针孔渲染范围：覆盖有效区域内所有像素去畸变后的方向
    mask = np.load(HERE / 'dk_params.npz')['mask']
    vs, us = np.nonzero(mask)
    pts = np.stack([us, vs], -1).astype(np.float64)
    xn = undistort(pts, K_d, dist)
    f_r = K_d[0, 0]
    half_w = int(np.ceil(np.abs(xn[:, 0]).max() * f_r)) + 4
    half_h = int(np.ceil(np.abs(xn[:, 1]).max() * f_r)) + 4
    Wr, Hr = 2 * half_w, 2 * half_h
    K_r = [[f_r, 0, half_w - 0.5], [0, f_r, half_h - 0.5], [0, 0, 1]]   # 光轴过图像正中

    scene = bpy.context.scene
    r, c = scene.render, scene.cycles
    r.use_multiview = False
    scene.use_nodes = False
    r.resolution_percentage = 100
    r.use_persistent_data = True
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == 'OPTIX'
    c.device = 'GPU'
    coll = bpy.data.collections.new('DK_Sim')
    scene.collection.children.link(coll)
    cam_c = bpy.data.objects.new('DK_Color', bpy.data.cameras.new('DK_Color'))
    cam_d = bpy.data.objects.new('DK_Depth', bpy.data.cameras.new('DK_Depth'))
    for o in (cam_c, cam_d):
        coll.objects.link(o)
    set_intrinsics(cam_c.data, K_c, Wc, Hc)
    set_intrinsics(cam_d.data, K_r, Wr, Hr)

    # 玻璃（透射）与高光金属物体的实例 ID，后处理按材质规则处理
    glass, metal = set(), set()
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        for s in o.material_slots:
            m = s.material
            if not (m and m.use_nodes):
                continue
            for n in m.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    if n.inputs['Transmission Weight'].default_value > 0.5:
                        glass.add(o)
                    elif n.inputs['Metallic'].default_value > 0.5 and n.inputs['Roughness'].default_value < 0.2:
                        metal.add(o)
                elif n.type == 'BSDF_GLASS':
                    glass.add(o)
    mirror_obj = {o for o in bpy.data.objects if o.type == 'MESH'
                  and any(surface_kind(s.material) == 'mirror' for s in o.material_slots)}

    rig = {'depth': {'K': K_d.tolist(), 'dist_k1k2p1p2k3k4k5k6': dist.tolist(), 'size': [Wd, Hd]},
           'color': {'K': K_c.tolist(), 'size': [Wc, Hc], 'distortion': 'none (pinhole render)'},
           'T_color_from_depth': T_c_d.tolist(), 'pinhole_render': {'K': K_r, 'size': [Wr, Hr]},
           'glass_ids': sorted({o.pass_index for o in glass}), 'metal_ids': sorted({o.pass_index for o in metal}),
           'mirror_ids': sorted({o.pass_index for o in mirror_obj}),
           'views': {}}
    poses = {}
    for name, v in views.items():
        M_c = look_at(v[:3], v[3:])
        T_w_c = np.array(M_c @ CV2BL)                       # 彩色相机 OpenCV 系 → 世界
        T_w_d = T_w_c @ np.linalg.inv(T_c_d)                # 深度相机 OpenCV 系 → 世界
        poses[name] = (M_c, Matrix(T_w_d.tolist()) @ CV2BL)
        rig['views'][name] = {'c2w_color_opencv': T_w_c.tolist(), 'c2w_depth_opencv': T_w_d.tolist()}
    (out / 'rig.json').write_text(json.dumps(rig, indent=1))
    if 'dataset_labels.json' in bpy.data.texts:
        (out / 'labels.json').write_text(bpy.data.texts['dataset_labels.json'].as_string())

    # 1) 彩色相机：正常灯光
    scene.camera = cam_c
    r.resolution_x, r.resolution_y = Wc, Hc
    c.samples = opt.color_samples
    c.use_denoising = True
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_depth = '8'
    r.image_settings.color_mode = 'RGB'
    for name, (M_c, _) in poses.items():
        (out / name).mkdir(exist_ok=True)
        cam_c.matrix_world = M_c
        r.filepath = str(out / name / 'color.png')
        bpy.ops.render.render(write_still=True)
        print('COLOR', name, flush=True)

    # 2) 深度相机：关掉所有灯、天空与自发光，只留相机处的红外点光源
    for o in bpy.data.objects:
        if o.type == 'LIGHT':
            o.hide_render = True
    if scene.world and scene.world.use_nodes:
        for n in scene.world.node_tree.nodes:
            if n.type == 'BACKGROUND':
                n.inputs['Strength'].default_value = 0
    for m in bpy.data.materials:
        if m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs['Emission Strength'].default_value = 0
                elif n.type == 'EMISSION':
                    n.inputs['Strength'].default_value = 0
    ir = bpy.data.objects.new('DK_IR', bpy.data.lights.new('DK_IR', 'POINT'))
    ir.data.energy = 100.0
    ir.data.shadow_soft_size = 0.005
    coll.objects.link(ir)
    ir.parent = cam_d
    scene.camera = cam_d
    r.resolution_x, r.resolution_y = Wr, Hr
    r.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    r.image_settings.color_depth = '32'
    r.image_settings.exr_codec = 'ZIP'
    tracer = MirrorTracer(scene)
    mirror_ids = np.array(rig['mirror_ids'])
    for name, (_, M_d) in poses.items():
        cam_d.matrix_world = M_d
        c.samples, c.use_denoising = opt.samples, True
        r.filepath = str(out / name / 'ir_raw.exr')
        bpy.ops.render.render(write_still=True)
        ch = read_exr(out / name / 'ir_raw.exr')
        light = {p: passes(ch, p, 'RGB').astype(np.float32) for p in
                 ('Combined', 'DiffCol', 'DiffDir', 'DiffInd', 'GlossCol', 'GlossDir', 'GlossInd',
                  'TransCol', 'TransDir', 'TransInd')}
        np.savez_compressed(out / name / 'ir_passes.npz', **light,
                            depth=passes(ch, 'Depth', 'Z')[..., 0], normal=passes(ch, 'Normal', 'XYZ').astype(np.float32),
                            index=passes(ch, 'IndexOB', 'X')[..., 0].astype(np.uint16))
        (out / name / 'ir_raw.exr').unlink()
        # 镜面：只追踪物体 ID 属于含镜面材质物体的像素（镜框与镜面同 ID，由追踪结果区分）
        mres = np.zeros((4, Hr, Wr), np.float32)
        T = np.array(rig['views'][name]['c2w_depth_opencv'])
        o_w = Vector(T[:3, 3])
        vs_m, us_m = np.nonzero(np.isin(passes(ch, 'IndexOB', 'X')[..., 0].astype(np.int32), mirror_ids))
        dirs = np.stack([(us_m - K_r[0][2]) / f_r, (vs_m - K_r[1][2]) / f_r, np.ones(len(us_m))], -1) @ T[:3, :3].T
        for v_, u_, d_ in zip(vs_m, us_m, dirs):
            res = tracer.trace(o_w, Vector(d_).normalized())
            if res:
                mres[:, v_, u_] = res
        np.savez_compressed(out / name / 'mirror.npz', range=mres[0], cos_mirror=mres[1], cos_end=mres[2],
                            bounces=mres[3].astype(np.uint8))
        print('MIRROR', name, int((mres[3] > 0).sum()), 'of', len(us_m), 'candidate px', flush=True)
        # 透过玻璃：隐去玻璃后只需深度，1 采样即可
        for o in glass:
            o.visible_camera = False
        c.samples, c.use_denoising = 1, False
        r.filepath = str(out / name / 'through.exr')
        bpy.ops.render.render(write_still=True)
        ch = read_exr(out / name / 'through.exr')
        np.savez_compressed(out / name / 'through.npz', depth=passes(ch, 'Depth', 'Z')[..., 0])
        (out / name / 'through.exr').unlink()
        for o in glass:
            o.visible_camera = True
        print('DEPTHCAM', name, flush=True)



if __name__ == '__main__':
    main()
