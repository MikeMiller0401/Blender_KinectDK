"""关灯路径低清预览：单目、低分辨率、低采样，按关灯时间表合成后出图，用于检查路线与关灯时机。
用法：blender -b apartment.blend --python preview_tour.py -- 输出目录 [--camera Cam_LightTour] [--step 3]
      [--percent 40] [--samples 32] [--range A-B]
之后用 ffmpeg 把 输出目录/*.png 合成视频。不保存源文件。
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_dataset as rd  # noqa: E402  复用三脚架跟随与关灯权重


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--camera', default='Cam_LightTour')
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--percent', type=int, default=40)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--range')
    opt = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    cam = scene.camera = bpy.data.objects[opt.camera]
    r, c = scene.render, scene.cycles
    r.use_multiview = False
    r.resolution_percentage = opt.percent
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_depth = '8'
    r.use_persistent_data = True
    c.samples, c.adaptive_min_samples = opt.samples, min(16, opt.samples)
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == 'OPTIX'
    c.device = 'GPU'

    switch = json.loads(cam['light_switch']) if 'light_switch' in cam else None
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    outs = {s.name: s for s in rl.outputs if s.enabled}
    total, mixers = None, []
    for g in (n for n in outs if n.startswith('Combined_')):
        sock = outs[g]
        if switch and g[len('Combined_'):] in switch['rooms']:
            mix = tree.nodes.new('CompositorNodeMixRGB')
            mix.blend_type = 'MIX'
            mix.inputs[1].default_value = (0, 0, 0, 1)
            tree.links.new(sock, mix.inputs[2])
            mixers.append(mix)
            sock = mix.outputs[0]
        if total is not None:
            add = tree.nodes.new('CompositorNodeMixRGB')
            add.blend_type = 'ADD'
            tree.links.new(total, add.inputs[1])
            tree.links.new(sock, add.inputs[2])
            sock = add.outputs[0]
        total = sock
    den = tree.nodes.new('CompositorNodeDenoise')
    tree.links.new(total, den.inputs['Image'])
    tree.links.new(outs['Denoising Normal'], den.inputs['Normal'])
    tree.links.new(outs['Denoising Albedo'], den.inputs['Albedo'])
    comp = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(den.outputs[0], comp.inputs['Image'])

    rig = rd.attach_rig(cam)
    a, b = cam['tour_range'] if 'tour_range' in cam else (scene.frame_start, scene.frame_end)
    if opt.range:
        a, b = (int(v) for v in opt.range.split('-'))
    frames = list(range(a, b + 1, opt.step))
    for i, f in enumerate(frames, 1):
        scene.frame_set(f)
        if rig:
            rd.follow_rig(cam, rig)
        for mix in mixers:
            mix.inputs[0].default_value = rd.light_weight(f, switch)
        r.filepath = str(out / f'tour_{i:04d}_f{f:04d}.png')
        bpy.ops.render.render(write_still=True)
        if i % 20 == 0 or i == len(frames):
            print(f'PROGRESS {i}/{len(frames)} 帧 {f}', flush=True)


if __name__ == '__main__':
    main()
