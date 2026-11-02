"""关灯路径关键帧拼图：上排为 A 起点（开灯）与 A 终点（关灯）同视角对比，下排为路线各关键时刻。
用系统 Python 运行（cv2 + PIL）：python3 lighttour_sheet.py 渲染输出目录 帧号,标题;帧号,标题;...
读取 scenario/（关灯场景）左目图像与 lighting.json，输出 渲染输出目录/lighttour_sheet.jpg。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def load(path, width):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = (img.astype(np.float32) / (65535 if img.dtype == np.uint16 else 255) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = round(img.shape[0] * width / img.shape[1])
    return Image.fromarray(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))


def main():
    out = Path(sys.argv[1])
    keys = [(int(k.split(',')[0]), k.split(',', 1)[1]) for k in sys.argv[2].split(';')]
    light = json.loads((out / 'lighting.json').read_text())
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17)
    first, last = keys[0], keys[-1]
    big = [load(out / 'scenario' / f'scenario_{f:04d}_L.png', 800) for f, _ in (first, last)]
    rest = keys[1:-1]
    tw = 1600 // len(rest)
    tiles = [load(out / 'scenario' / f'scenario_{f:04d}_L.png', tw) for f, _ in rest]
    pad = 44
    sheet = Image.new('RGB', (1600, pad + big[0].height + pad + tiles[0].height + 70), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    rooms = ', '.join(light['rooms_off'])
    titles = [f'{first[1]}  (frame {first[0]}, lights ON)', f'{last[1]}  (frame {last[0]}, {rooms} lights OFF)']
    for i, (img, t) in enumerate(zip(big, titles)):
        sheet.paste(img, (i * 800, pad))
        draw.text((i * 800 + 10, 10), t, font=font, fill=(240, 240, 240))
    y = pad + big[0].height + pad
    for i, ((f, t), img) in enumerate(zip(rest, tiles)):
        sheet.paste(img, (i * tw, y))
        state = 'ON' if light['weights'][str(f)] >= 1 else ('OFF' if light['weights'][str(f)] <= 0 else 'fading')
        draw.text((i * tw + 8, y - 30), f'{t} · f{f} · {rooms} {state}', font=small, fill=(220, 220, 220))
    note = (f'Path A (Living) → B (Bedroom1) → C (Bedroom2) → A. {rooms} lights switch off at frame '
            f'{light["switch_frame"]} while the camera faces into C; start and end frames share the same pose.')
    draw.text((10, y + tiles[0].height + 12), note, font=small, fill=(200, 200, 200))
    draw.text((10, y + tiles[0].height + 38),
              'OFF image = sum of per-room light groups without the switched-off room, then denoised '
              '(exact by linearity of light transport).', font=small, fill=(200, 200, 200))
    sheet.save(out / 'lighttour_sheet.jpg', quality=92)
    print('SHEET', out / 'lighttour_sheet.jpg')


if __name__ == '__main__':
    main()
