"""数据集后处理：由实例图生成语义图，并可为指定帧生成预览拼图。
用 Blender 自带 Python 运行（含 numpy 与 OpenImageIO）：
  blender -b --factory-startup --python postprocess_dataset.py -- 输出目录 [--preview]
语义图（类别表与取值约定见 labels.json）：
  semantic/            8-bit 自定义 30 类 ID（0 = 空/机位道具）
  semantic_scannet200/ 16-bit ScanNet 原始类别 ID（0 = 未标注）
  semantic_nyu40/      8-bit NYU40 ID（0 = 未标注）
  semantic_scannet20/  8-bit 训练索引 0-19（255 = 忽略/未标注）
--preview：每帧生成 preview/preview_<帧号>.jpg（RGB 左右、深度、法线、albedo、shading、语义、实例、各房间灯光）
Blender 自带的 OpenImageIO 不支持文字渲染，标签由系统 Python 的 PIL 补写（本文件 --label 模式）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

TILE_W, TILE_H = 533, 333
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def legend(baseline_cm):
    # 预览图底部的说明文字
    return [
        f'Top: left / right RGB (parallel stereo, {baseline_cm:.0f} cm baseline) · Z-depth to camera plane '
        '(near = yellow, far = purple) · world-space normals',
        'Middle: albedo (diffuse colour) · diffuse shading (lighting only, no texture) · semantic class '
        '(ScanNet200; NYU40 / ScanNet20 maps also exported) · instance (random colours)',
        "Bottom: light emitted by each room's fixtures (Sky = daylight through windows), including bounces "
        'and mirror reflections; the 4 largest groups are shown',
        '% = pixel-mean linear radiance of the group / sum of all 8 groups (all groups add up to 100%). '
        'E.g. 12% means switching those lights off darkens the image by about 12%',
        'Each light tile is exposed separately to show where its light lands: tile brightness does not '
        'reflect the share, and heavy grain means the light is weak',
    ]


def read(path):
    inp = oiio.ImageInput.open(str(path))
    spec = inp.spec()
    pixels = np.asarray(inp.read_image(oiio.FLOAT))
    names = list(spec.channelnames)
    inp.close()
    return pixels, names


def write(path, arr):
    arr = np.ascontiguousarray(arr)
    fmt = oiio.UINT16 if arr.dtype == np.uint16 else oiio.UINT8
    spec = oiio.ImageSpec(arr.shape[1], arr.shape[0], arr.shape[2] if arr.ndim == 3 else 1, fmt)
    out = oiio.ImageOutput.create(str(path))
    out.open(str(path), spec)
    out.write_image(arr)
    out.close()


def palette(n, seed):
    rng = np.random.default_rng(seed)
    pal = rng.integers(40, 255, size=(n, 3)).astype(np.float32) / 255
    pal[0] = 0
    return pal


def tonemap(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def colormap(t):
    # 近似 viridis：低值深紫、高值黄
    anchors = np.array([[0.27, 0.00, 0.33], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55],
                        [0.37, 0.79, 0.38], [0.99, 0.91, 0.14]])
    t = np.clip(t, 0, 1) * (len(anchors) - 1)
    i = np.minimum(t.astype(int), len(anchors) - 2)
    f = (t - i)[..., None]
    return anchors[i] * (1 - f) + anchors[i + 1] * f


def tile(img, label, labels):
    buf = oiio.ImageBuf(np.ascontiguousarray(img[..., :3].astype(np.float32)))
    small = oiio.ImageBufAlgo.resize(buf, roi=oiio.ROI(0, TILE_W, 0, TILE_H, 0, 1, 0, 3))
    labels.append(label)
    return np.asarray(small.get_pixels(oiio.FLOAT))


def draw_labels(sheet_path, labels, notes):
    from PIL import Image, ImageDraw, ImageFont
    tiles = Image.open(sheet_path)
    line_h, pad = 27, 14
    img = Image.new('RGB', (tiles.width, tiles.height + pad * 2 + line_h * len(notes)), (24, 24, 24))
    img.paste(tiles, (0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 17)
    for i, text in enumerate(labels):
        x, y = (i % 4) * TILE_W, (i // 4) * TILE_H
        w = draw.textlength(text, font=font)
        draw.rectangle([x, y, x + w + 12, y + 26], fill=(0, 0, 0))
        draw.text((x + 6, y + 4), text, font=font, fill=(255, 255, 255))
    for i, text in enumerate(notes):
        draw.text((pad, tiles.height + pad + i * line_h), text, font=font, fill=(235, 235, 235))
    img.save(sheet_path, quality=92)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    out = Path(argv[0])
    labels = json.loads((out / 'labels.json').read_text())
    # 实例 ID → 各套类别的查找表；实例 0（背景/机位道具）为未标注
    schemes = {'semantic': ('class_id', np.uint8, 0), 'semantic_scannet200': ('scannet200_id', np.uint16, 0),
               'semantic_nyu40': ('nyu40_id', np.uint8, 0), 'semantic_scannet20': ('scannet20_id', np.uint8, 255)}
    luts = {}
    for folder, (field, dtype, empty) in schemes.items():
        luts[folder] = np.full(65536, empty, dtype)
        for iid, info in labels['instances'].items():
            luts[folder][int(iid)] = info[field]
        (out / folder).mkdir(exist_ok=True)
    lut = luts['semantic']
    s200_ids = labels['schemes']['scannet200']['ids']
    s200_pal = np.zeros((max(s200_ids) + 1, 3), np.float32)
    s200_pal[s200_ids] = palette(len(s200_ids) + 1, 7)[1:]
    inst_pal = palette(65536, 11)
    for p in sorted((out / 'instance').glob('instance_*.png')):
        ids = np.round(read(p)[0][..., 0] * 65535).astype(np.int64)
        for folder, table in luts.items():
            write(out / folder / p.name.replace('instance_', 'semantic_'), table[ids][..., None])
    print('SEMANTIC', {f: len(list((out / f).glob('*.png'))) for f in schemes}, flush=True)
    if '--preview' not in argv:
        return

    (out / 'preview').mkdir(exist_ok=True)
    cams = json.loads((out / 'cameras.json').read_text())
    for fr in cams['frames']:
        f = fr['frame']
        rgb_l, _ = read(out / fr['views']['L']['image'])
        rgb_r, _ = read(out / fr['views']['R']['image'])
        depth = read(out / 'depth' / f'depth_{f:04d}_L.exr')[0][..., 0]
        valid = depth < 1e5
        lo, hi = np.percentile(depth[valid], [1, 99])
        depth_vis = colormap(1 - (depth - lo) / (hi - lo + 1e-6)) * valid[..., None]
        normal = read(out / 'normal' / f'normal_{f:04d}_L.exr')[0][..., :3]
        albedo = read(out / 'albedo' / f'albedo_{f:04d}_L.png')[0]
        shading = tonemap(read(out / 'shading' / f'shading_{f:04d}_L.exr')[0][..., :3])
        ids = np.round(read(out / 'instance' / f'instance_{f:04d}_L.png')[0][..., 0] * 65535).astype(np.int64)
        names = []
        tiles = [tile(rgb_l, f'RGB left  f{f}', names), tile(rgb_r, 'RGB right', names),
                 tile(depth_vis, f'depth {lo:.1f}-{hi:.1f} m', names),
                 tile(normal * 0.5 + 0.5, 'normal (world)', names),
                 tile(albedo, 'albedo', names), tile(shading, 'diffuse shading', names),
                 tile(s200_pal[luts['semantic_scannet200'][ids]], 'semantic (ScanNet200)', names),
                 tile(inst_pal[ids], 'instance', names)]
        passes_path = out / 'passes' / f'passes_{f:04d}_L.exr'
        if passes_path.exists():
            p, channels = read(passes_path)
            groups = sorted({n.rsplit('.', 1)[0] for n in channels if n.startswith('Combined_')})
            lg = {g[9:]: p[..., [channels.index(f'{g}.{c}') for c in 'RGB']] for g in groups}
            total = sum(lg.values()).mean() + 1e-9
            # 按贡献从大到小取前 4 个房间；每块单独曝光以看清照亮范围，标签给出占总光照的比例
            for name, img in sorted(lg.items(), key=lambda kv: -kv[1].mean())[:4]:
                gain = 0.6 / (np.percentile(img.mean(axis=-1), 95) + 1e-6)
                tiles.append(tile(tonemap(img * gain), f'light {name} {img.mean() / total:.1%}', names))
        rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
        sheet = np.concatenate(rows, axis=0)
        sheet_path = out / 'preview' / f'preview_{f:04d}.jpg'
        write(sheet_path, (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        notes = legend((cams.get('baseline_m') or 0) * 100)
        if len(tiles) <= 8:
            notes = notes[:2]
        subprocess.run(['/usr/bin/python3', __file__, '--label', str(sheet_path), json.dumps(names),
                        json.dumps(notes, ensure_ascii=False)], check=True, env=env)
        print('PREVIEW', f, flush=True)


if __name__ == '__main__':
    if '--label' in sys.argv:
        i = sys.argv.index('--label')
        draw_labels(sys.argv[i + 1], json.loads(sys.argv[i + 2]), json.loads(sys.argv[i + 3]))
    else:
        import numpy as np
        import OpenImageIO as oiio
        main()
