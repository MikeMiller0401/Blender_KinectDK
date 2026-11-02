"""数据集渲染：双目 RGB + 深度/法线/albedo/shading/实例 + 原始光照通道与房间灯光分组；可选关灯场景合成。
用法：blender -b apartment.blend --python render_dataset.py -- 输出目录
        [--camera 名称] (--step N | --frames 1,2,3 | --range A-B) [--no-passes]
        [--lights-off Living,Kitchen --switch-frame F --fade N]
相机若带自定义属性 tour_range / light_switch（见 lighttour_setup.py），未指定帧或关灯参数时以其为默认。
非 Cam_MirrorTour 相机渲染时，三脚架道具按固定几何关系逐帧跟随当前相机（仅内存中，不保存）。
渲染设置、灯光分组、实例 ID、双目参数均取自 blend（由 dataset_setup.py 写入）；不保存源文件。
每帧每眼（_L/_R）输出：
  frames/   16-bit PNG 最终图（已降噪）
  depth/    32-bit EXR，到相机平面的 Z 深度（米）
  normal/   16-bit EXR，世界空间法线
  albedo/   16-bit PNG，漫反射颜色（Standard sRGB）
  shading/  16-bit EXR，漫反射光照 DiffDir+DiffInd（线性，已降噪）
  instance/ 16-bit PNG，实例 ID（=pass_index，对照 labels.json）
  passes/   多层 EXR：全部原始光照通道 + 各房间灯光分组 Combined_<房间>（未降噪）
  scenario/ 16-bit PNG，关灯场景：按时间表对关灯房间的灯光分组加权后求和再降噪（仅设置关灯时输出）；
            frames/ 始终为全部开灯。权重与时间表见 lighting.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

LIGHT_PASSES = ['DiffCol', 'DiffDir', 'DiffInd', 'GlossCol', 'GlossDir', 'GlossInd',
                'TransCol', 'TransDir', 'TransInd', 'Emit', 'Env']


RIG_PIVOT = Vector((0, -0.0751, 0.1276))   # 云台枢轴在相机局部坐标中的位置（由 MirrorTour 烘焙数据反推）


def attach_rig(cam):
    """让三脚架道具跟随非 MirrorTour 相机：解除其烘焙动画，相机部件改挂到当前相机。"""
    objs = bpy.data.objects
    if cam.name == 'Cam_MirrorTour' or 'Cam_MirrorTour_Rig' not in objs:
        return None
    parts = [objs[n] for n in ('Cam_MirrorTour_Dolly', 'Cam_MirrorTour_Head', 'Cam_MirrorTour_Column')]
    for o in parts:
        if o.animation_data:
            o.animation_data.action = None
    rig = objs['Cam_MirrorTour_Rig']
    rig.parent = cam
    rig.matrix_parent_inverse = Matrix.Identity(4)
    return parts


def follow_rig(cam, parts):
    """按 MirrorTour 的几何关系摆放滑轮、云台与中轴：枢轴在镜头后下方，滑轮在其正下方。"""
    dolly, head, column = parts
    M = cam.matrix_world
    pivot = M @ RIG_PIVOT
    fwd = M.to_3x3() @ Vector((0, 0, -1))
    yaw = math.atan2(fwd.y, fwd.x)
    dolly.location = (pivot.x, pivot.y, 0.0)
    head.location = pivot
    head.rotation_euler = (0.0, 0.0, yaw - math.pi / 2)
    column.location = (pivot.x, pivot.y, 1.0)
    column.scale = (1.0, 1.0, pivot.z - 1.075)


def light_weight(frame, switch):
    """关灯房间在该帧的灯光权重：切换前 1，之后线性降到 0（fade_frames=0 即瞬间关灯）。"""
    if not switch or frame < switch['frame']:
        return 1.0
    fade = switch.get('fade_frames', 0)
    return 0.0 if fade <= 0 else max(0.0, 1.0 - (frame - switch['frame']) / fade)


def intrinsics(scene, cam):
    r = scene.render
    w = r.resolution_x * r.resolution_percentage / 100
    h = r.resolution_y * r.resolution_percentage / 100
    d = cam.data
    fit = d.sensor_fit if d.sensor_fit != 'AUTO' else ('HORIZONTAL' if w >= h else 'VERTICAL')
    sensor = d.sensor_width if fit == 'HORIZONTAL' else d.sensor_height
    f = d.lens / sensor * (w if fit == 'HORIZONTAL' else h)
    return [[f, 0, w / 2 - d.shift_x * max(w, h)], [0, f, h / 2 + d.shift_y * max(w, h)], [0, 0, 1]]


def camera_record(scene, cam, index, frame, stem, views):
    K = intrinsics(scene, cam)
    rec = {'index': index, 'frame': frame, 'K': K, 'views': {}}
    c2w = cam.matrix_world.copy()
    for v in views:
        # 平行双目、以左眼为基准：右眼沿相机局部 +X 平移基线
        offset = cam.data.stereo.interocular_distance if v == 'R' else 0.0
        c2w_v = c2w @ Matrix.Translation((offset, 0, 0))
        c2w_cv = c2w_v @ Matrix.Diagonal((1, -1, -1, 1))
        rec['views'][v] = {
            'image': f'frames/{stem}_{v}.png',
            'c2w_blender': [list(row) for row in c2w_v],
            'w2c_opencv': [list(row) for row in c2w_cv.inverted()],
        }
    return rec


def build_compositor(scene, out, with_passes, switch=None):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    outs = {s.name: s for s in rl.outputs if s.enabled}
    need = ['Image', 'Depth', 'Normal', 'IndexOB', 'Denoising Normal'] + LIGHT_PASSES
    missing = [n for n in need if n not in outs]
    if missing:
        raise RuntimeError(f'缺少渲染通道 {missing}；请先运行 dataset_setup.py。现有：{list(outs)}')
    link = tree.links.new
    comp = tree.nodes.new('CompositorNodeComposite')
    link(outs['Image'], comp.inputs['Image'])

    def file_out(name, socket, fmt, mode, depth, raw=False, codec='ZIP', view=None):
        node = tree.nodes.new('CompositorNodeOutputFile')
        node.base_path = str(out / name)
        node.file_slots[0].path = f'{name}_'
        f = node.format
        f.file_format, f.color_mode, f.color_depth = fmt, mode, depth
        if fmt == 'OPEN_EXR':
            f.exr_codec = codec
        if view:
            f.color_management = 'OVERRIDE'
            f.view_settings.view_transform = view
            f.view_settings.look = 'None'
        if raw:
            node.file_slots[0].save_as_render = False
        link(socket, node.inputs[0])
        return node

    file_out('depth', outs['Depth'], 'OPEN_EXR', 'BW', '32')
    file_out('normal', outs['Normal'], 'OPEN_EXR', 'RGB', '16')
    file_out('albedo', outs['DiffCol'], 'PNG', 'RGB', '16', view='Standard')

    add = tree.nodes.new('CompositorNodeMixRGB')
    add.blend_type = 'ADD'
    add.inputs[0].default_value = 1.0
    link(outs['DiffDir'], add.inputs[1])
    link(outs['DiffInd'], add.inputs[2])
    # shading 本身不含纹理，albedo 辅助通道给纯白，只借法线引导降噪
    white = tree.nodes.new('CompositorNodeRGB')
    white.outputs[0].default_value = (1, 1, 1, 1)
    den = tree.nodes.new('CompositorNodeDenoise')
    den.prefilter = 'ACCURATE'
    link(add.outputs[0], den.inputs['Image'])
    link(outs['Denoising Normal'], den.inputs['Normal'])
    link(white.outputs[0], den.inputs['Albedo'])
    file_out('shading', den.outputs[0], 'OPEN_EXR', 'RGB', '16')

    # 实例 ID 为整数，按 1/65535 缩放写入 16-bit PNG，读取时乘回
    div = tree.nodes.new('CompositorNodeMath')
    div.operation = 'DIVIDE'
    div.inputs[1].default_value = 65535.0
    link(outs['IndexOB'], div.inputs[0])
    file_out('instance', div.outputs[0], 'PNG', 'BW', '16', raw=True)

    groups = [n for n in outs if n.startswith('Combined_')]
    mixers = []
    if switch:
        # 各房间灯光分组加权求和（关灯房间经 MIX 节点乘以权重），再用 albedo/法线引导降噪
        missing = [r for r in switch['rooms'] if f'Combined_{r}' not in outs]
        if missing or 'Denoising Albedo' not in outs:
            raise RuntimeError(f'关灯合成缺少灯光分组 {missing} 或降噪通道；请先运行 dataset_setup.py')
        total = None
        for g in groups:
            sock = outs[g]
            if g[len('Combined_'):] in switch['rooms']:
                mix = tree.nodes.new('CompositorNodeMixRGB')
                mix.blend_type = 'MIX'
                mix.inputs[0].default_value = 1.0
                mix.inputs[1].default_value = (0, 0, 0, 1)
                link(sock, mix.inputs[2])
                mixers.append(mix)
                sock = mix.outputs[0]
            if total is None:
                total = sock
                continue
            add_g = tree.nodes.new('CompositorNodeMixRGB')
            add_g.blend_type = 'ADD'
            add_g.inputs[0].default_value = 1.0
            link(total, add_g.inputs[1])
            link(sock, add_g.inputs[2])
            total = add_g.outputs[0]
        den_s = tree.nodes.new('CompositorNodeDenoise')
        den_s.prefilter = 'ACCURATE'
        link(total, den_s.inputs['Image'])
        link(outs['Denoising Normal'], den_s.inputs['Normal'])
        link(outs['Denoising Albedo'], den_s.inputs['Albedo'])
        file_out('scenario', den_s.outputs[0], 'PNG', 'RGB', '16')

    if with_passes:
        node = tree.nodes.new('CompositorNodeOutputFile')
        node.base_path = str(out / 'passes' / 'passes_')
        node.format.file_format = 'OPEN_EXR_MULTILAYER'
        node.format.color_depth = '16'
        node.format.exr_codec = 'PIZ'
        node.layer_slots.clear()
        for name in LIGHT_PASSES + groups:
            node.layer_slots.new(name)
            link(outs[name], node.inputs[name])
    return (groups if with_passes else []), mixers


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--step', type=int, default=10)
    ap.add_argument('--frames')
    ap.add_argument('--range')
    ap.add_argument('--camera')
    ap.add_argument('--no-passes', action='store_true')
    ap.add_argument('--lights-off', help='关灯房间，逗号分隔；none 表示不做关灯合成')
    ap.add_argument('--switch-frame', type=int)
    ap.add_argument('--fade', type=int, default=0, help='关灯渐变帧数，0 为瞬间关灯')
    opt = ap.parse_args(argv)
    out = Path(opt.out).resolve()
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if opt.camera:
        scene.camera = bpy.data.objects[opt.camera]
    cam = scene.camera
    if scene.render.engine != 'CYCLES' or cam is None:
        raise RuntimeError('需要 Cycles 引擎和活动相机')

    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    devices = [d.name for d in prefs.devices if d.use]
    if not devices:
        raise RuntimeError('未找到可用的 OptiX GPU')
    scene.cycles.device = 'GPU'
    scene.render.use_persistent_data = True
    stereo = scene.render.use_multiview and scene.render.views_format == 'STEREO_3D'
    views = ['L', 'R'] if stereo else ['']
    switch = json.loads(cam['light_switch']) if 'light_switch' in cam else None
    if opt.lights_off:
        switch = None if opt.lights_off == 'none' else {
            'rooms': opt.lights_off.split(','), 'frame': opt.switch_frame, 'fade_frames': opt.fade}
        if switch and switch['frame'] is None:
            raise SystemExit('--lights-off 需要同时给出 --switch-frame')
    groups, mixers = build_compositor(scene, out, not opt.no_passes, switch)
    rig_parts = attach_rig(cam)

    first, last = (cam['tour_range'] if 'tour_range' in cam else (scene.frame_start, scene.frame_end))
    if opt.frames:
        frames = sorted(dict.fromkeys(int(f) for f in opt.frames.split(',')))
    elif opt.range:
        a, b = (int(v) for v in opt.range.split('-'))
        frames = list(range(a, b + 1, opt.step))
    else:
        frames = list(range(first, last + 1, scene.frame_step * opt.step))
    blend_stem = Path(bpy.data.filepath).stem
    c = scene.cycles
    started = time.time()
    status = {
        'state': 'rendering', 'source': bpy.data.filepath, 'blender_version': bpy.app.version_string,
        'devices': devices, 'camera': cam.name, 'frames': frames, 'completed': 0,
        'stereo': stereo, 'baseline_m': cam.data.stereo.interocular_distance if stereo else None,
        'resolution': [scene.render.resolution_x, scene.render.resolution_y],
        'resolution_percentage': scene.render.resolution_percentage,
        'samples': c.samples, 'adaptive_threshold': c.adaptive_threshold,
        'adaptive_min_samples': c.adaptive_min_samples, 'denoiser': c.denoiser,
        'clamp_indirect': c.sample_clamp_indirect, 'animated_seed': c.use_animated_seed,
        'bounces': {'total': c.max_bounces, 'diffuse': c.diffuse_bounces, 'glossy': c.glossy_bounces,
                    'transmission': c.transmission_bounces},
        'view_transform': scene.view_settings.view_transform, 'look': scene.view_settings.look,
        'motion_blur': scene.render.use_motion_blur, 'dof': cam.data.dof.use_dof,
        'light_groups': [g[len('Combined_'):] for g in groups], 'light_switch': switch,
        'started_unix': started,
    }
    status_path = out / 'render_status.json'

    def update(**changes):
        status.update(changes, elapsed_seconds=round(time.time() - started, 1))
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(status_path)

    labels = bpy.data.texts.get('dataset_labels.json')
    if labels is None:
        raise RuntimeError('缺少 dataset_labels.json，请先运行 dataset_setup.py')
    (out / 'labels.json').write_text(labels.as_string(), encoding='utf-8')

    print(f'GPU: {devices}; 相机 {cam.name}; 双目 {stereo}; 共 {len(frames)} 帧', flush=True)
    cameras = []
    for index, frame in enumerate(frames, 1):
        stem = f'{blend_stem}_{index:04d}_f{frame:04d}'
        scene.frame_set(frame)
        if rig_parts:
            follow_rig(cam, rig_parts)
        for mix in mixers:
            mix.inputs[0].default_value = light_weight(frame, switch)
        cameras.append(camera_record(scene, cam, index, frame, stem, views))
        targets = [out / 'frames' / (f'{stem}_{v}.png' if v else f'{stem}.png') for v in views]
        if all(p.exists() and p.stat().st_size > 0 for p in targets):
            update(completed=index)
            continue
        scene.render.filepath = str(out / 'frames' / f'{stem}.png')
        t = time.time()
        bpy.ops.render.render(write_still=True)
        for p in targets:
            with p.open('rb') as stream:
                if stream.read(8) != b'\x89PNG\r\n\x1a\n':
                    raise RuntimeError(f'无效的 PNG: {p}')
        update(completed=index)
        elapsed = time.time() - started
        eta = elapsed / index * (len(frames) - index)
        print(f'PROGRESS {index}/{len(frames)} 帧 {frame} 用时 {time.time() - t:.1f}s '
              f'已用 {elapsed / 60:.1f} 分 预计剩余 {eta / 60:.1f} 分', flush=True)

    (out / 'cameras.json').write_text(json.dumps({
        'convention': 'K 为像素内参（左右眼相同）；c2w_blender 为 Blender 相机到世界（-Z 前、Y 上）；'
                      'w2c_opencv 为世界到 OpenCV 相机（+Z 前、Y 下）；depth 为到相机平面的 Z 深度；单位：米。'
                      '双目为平行光轴，右眼 = 左眼沿相机 +X 平移 baseline_m',
        'width': scene.render.resolution_x * scene.render.resolution_percentage // 100,
        'height': scene.render.resolution_y * scene.render.resolution_percentage // 100,
        'baseline_m': status['baseline_m'], 'frames': cameras}, indent=1), encoding='utf-8')
    if switch:
        (out / 'lighting.json').write_text(json.dumps({
            'note': 'scenario/ 为关灯场景（按 weights 对关灯房间的灯光分组加权）；frames/ 始终全部开灯。'
                    '同一帧 frames/ 与 scenario/ 视角相同，可作开/关灯配对',
            'rooms_off': switch['rooms'], 'switch_frame': switch['frame'],
            'fade_frames': switch.get('fade_frames', 0),
            'weights': {str(f): light_weight(f, switch) for f in frames}}, indent=1), encoding='utf-8')
    update(state='rendered')


if __name__ == '__main__':
    main()
