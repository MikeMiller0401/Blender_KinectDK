"""由 render_sampled.sh 调用：按步长抽帧、100% 分辨率、全部 NVIDIA GPU 渲染；不保存源 blend。"""
import json
import sys
import time
from pathlib import Path

import bpy


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    out, step = Path(args[0]).resolve(), int(args[1])
    frames_dir = out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if len(args) > 2 and args[2]:
        # 仅本次渲染改用指定相机，不保存到源文件
        camera = bpy.data.objects.get(args[2])
        if camera is None or camera.type != 'CAMERA':
            raise RuntimeError(f'找不到相机: {args[2]}')
        scene.camera = camera
    if scene.render.engine != 'CYCLES' or scene.camera is None:
        raise RuntimeError('需要 Cycles 引擎和活动相机')

    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    devices = [d.name for d in prefs.devices if d.use]
    if not devices:
        raise RuntimeError('未找到可用的 OptiX GPU')

    original_percentage = scene.render.resolution_percentage
    scene.cycles.device = 'GPU'
    scene.render.resolution_percentage = 100
    # 帧间保留场景数据，避免每帧重新加载
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'PNG'
    frames = list(range(scene.frame_start, scene.frame_end + 1, scene.frame_step * step))
    stem = Path(bpy.data.filepath).stem
    started = time.time()
    status_path = out / 'render_status.json'
    status = {
        'state': 'rendering', 'source': bpy.data.filepath, 'scene': scene.name,
        'blender_version': bpy.app.version_string, 'devices': devices, 'camera': scene.camera.name,
        'frame_step': step, 'frames': frames, 'completed': 0,
        'source_fps': scene.render.fps / scene.render.fps_base,
        'resolution': [scene.render.resolution_x, scene.render.resolution_y],
        'original_resolution_percentage': original_percentage, 'resolution_percentage': 100,
        'samples': scene.cycles.samples, 'denoising': scene.cycles.use_denoising,
        'adaptive_threshold': scene.cycles.adaptive_threshold, 'started_unix': started,
    }

    def update(**changes):
        status.update(changes, elapsed_seconds=round(time.time() - started, 1))
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(status_path)

    print(f'GPU: {devices}; 相机 {scene.camera.name}; 共 {len(frames)} 帧（每 {step} 帧取 1 帧）', flush=True)
    for index, frame in enumerate(frames, 1):
        # 序号连续命名，便于 ffmpeg 直接合成；原始帧号记在 render_status.json
        path = frames_dir / f'{stem}_{index:04d}_f{frame:04d}.png'
        if path.exists() and path.stat().st_size > 0:
            update(completed=index)
            continue
        scene.frame_set(frame)
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
    update(state='rendered')


if __name__ == '__main__':
    main()
