"""由 render_vggt.sh 调用：按步长抽帧渲染 VGGT 数据集，使用文件内保存的渲染设置；不保存源 blend。
每帧输出：frames/ 16-bit PNG，depth/ 32-bit EXR（Cycles Depth 通道，即到相机平面的 Z 深度，单位米），
并把每帧相机内外参写入 cameras.json。"""
import json
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix


def camera_record(scene, cam, index, frame, image):
    r = scene.render
    w = r.resolution_x * r.resolution_percentage / 100
    h = r.resolution_y * r.resolution_percentage / 100
    d = cam.data
    sensor_fit = d.sensor_fit if d.sensor_fit != 'AUTO' else ('HORIZONTAL' if w >= h else 'VERTICAL')
    sensor = d.sensor_width if sensor_fit == 'HORIZONTAL' else d.sensor_height
    f = d.lens / sensor * (w if sensor_fit == 'HORIZONTAL' else h)
    cx = w / 2 - d.shift_x * max(w, h)
    cy = h / 2 + d.shift_y * max(w, h)
    c2w = cam.matrix_world.copy()
    # Blender 相机看向 -Z、Y 向上；OpenCV 看向 +Z、Y 向下
    c2w_cv = c2w @ Matrix.Diagonal((1, -1, -1, 1))
    return {
        'index': index, 'frame': frame, 'image': image,
        'K': [[f, 0, cx], [0, f, cy], [0, 0, 1]],
        'c2w_blender': [list(row) for row in c2w],
        'w2c_opencv': [list(row) for row in c2w_cv.inverted()],
    }


def add_pass_outputs(scene, out):
    scene.use_nodes = True
    tree = scene.node_tree
    layers = next((n for n in tree.nodes if n.type == 'R_LAYERS'), None) or tree.nodes.new('CompositorNodeRLayers')
    composite = next((n for n in tree.nodes if n.type == 'COMPOSITE'), None) or tree.nodes.new('CompositorNodeComposite')
    if not composite.inputs['Image'].is_linked:
        tree.links.new(layers.outputs['Image'], composite.inputs['Image'])
    for name, socket, mode, depth in (('depth', 'Depth', 'BW', '32'),):
        node = tree.nodes.new('CompositorNodeOutputFile')
        node.base_path = str(out / name)
        node.file_slots[0].path = f'{name}_'
        node.format.file_format = 'OPEN_EXR'
        node.format.color_mode = mode
        node.format.color_depth = depth
        node.format.exr_codec = 'ZIP'
        tree.links.new(layers.outputs[socket], node.inputs[0])


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    out, step = Path(args[0]).resolve(), int(args[1])
    frames_dir = out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
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
    # blend 已开启双目（dataset_setup.py），本脚本只出单目
    scene.render.use_multiview = False
    add_pass_outputs(scene, out)

    frames = list(range(scene.frame_start, scene.frame_end + 1, scene.frame_step * step))
    stem = Path(bpy.data.filepath).stem
    c = scene.cycles
    started = time.time()
    status_path = out / 'render_status.json'
    status = {
        'state': 'rendering', 'source': bpy.data.filepath, 'scene': scene.name,
        'blender_version': bpy.app.version_string, 'devices': devices, 'camera': cam.name,
        'frame_step': step, 'frames': frames, 'completed': 0,
        'source_fps': scene.render.fps / scene.render.fps_base,
        'resolution': [scene.render.resolution_x, scene.render.resolution_y],
        'resolution_percentage': scene.render.resolution_percentage,
        'samples': c.samples, 'adaptive_threshold': c.adaptive_threshold,
        'adaptive_min_samples': c.adaptive_min_samples, 'denoiser': c.denoiser,
        'clamp_indirect': c.sample_clamp_indirect, 'animated_seed': c.use_animated_seed,
        'bounces': {'total': c.max_bounces, 'diffuse': c.diffuse_bounces, 'glossy': c.glossy_bounces,
                    'transmission': c.transmission_bounces},
        'view_transform': scene.view_settings.view_transform, 'look': scene.view_settings.look,
        'motion_blur': scene.render.use_motion_blur, 'dof': cam.data.dof.use_dof,
        'started_unix': started,
    }

    def update(**changes):
        status.update(changes, elapsed_seconds=round(time.time() - started, 1))
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(status_path)

    print(f'GPU: {devices}; 相机 {cam.name}; 共 {len(frames)} 帧（每 {step} 帧取 1 帧）', flush=True)
    cameras = []
    for index, frame in enumerate(frames, 1):
        name = f'{stem}_{index:04d}_f{frame:04d}.png'
        path = frames_dir / name
        scene.frame_set(frame)
        cameras.append(camera_record(scene, cam, index, frame, f'frames/{name}'))
        if path.exists() and path.stat().st_size > 0 and (out / 'depth' / f'depth_{frame:04d}.exr').exists():
            update(completed=index)
            continue
        scene.render.filepath = str(path)
        t = time.time()
        bpy.ops.render.render(write_still=True)
        with path.open('rb') as stream:
            if stream.read(8) != b'\x89PNG\r\n\x1a\n':
                raise RuntimeError(f'无效的 PNG: {path}')
        update(completed=index)
        elapsed = time.time() - started
        eta = elapsed / index * (len(frames) - index)
        print(f'PROGRESS {index}/{len(frames)} 帧 {frame} 用时 {time.time() - t:.1f}s '
              f'已用 {elapsed / 60:.1f} 分 预计剩余 {eta / 60:.1f} 分', flush=True)
    (out / 'cameras.json').write_text(json.dumps({
        'convention': 'K 为像素内参；depth/depth_<帧号>.exr 为到相机平面的 Z 深度（非射线距离）；c2w_blender 为 Blender 相机到世界（-Z 前、Y 上）；'
                      'w2c_opencv 为世界到 OpenCV 相机（+Z 前、Y 下）。单位：米',
        'width': scene.render.resolution_x * scene.render.resolution_percentage // 100,
        'height': scene.render.resolution_y * scene.render.resolution_percentage // 100,
        'frames': cameras}, indent=1), encoding='utf-8')
    update(state='rendered')


if __name__ == '__main__':
    main()
