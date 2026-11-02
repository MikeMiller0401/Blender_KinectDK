#!/usr/bin/env bash
# VGGT 数据集：抽帧渲染（RGB+深度+相机参数）→ 合成视频 → 分包压缩 → 上传百度网盘 → QQ 发视频
# 用法：render_vggt.sh [输出目录]   重跑同一目录会跳过已渲染的帧；渲染设置取自 apartment.blend
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../apartment.blend"
STEP=10
OUT=${1:-"$HERE/vggt_every${STEP}_$(date +%Y%m%d_%H%M)"}
NAME=apartment_vggt_every${STEP}_$(basename "$OUT" | grep -oE '[0-9]{8}' | head -1)
REMOTE=/UbuntuData/Renders/apartment/$(basename "$OUT")
BLENDER=/home/tcy/Applications/Blender4.5.12/blender
mkdir -p "$OUT"
echo "输出目录：$OUT"

"$BLENDER" -b "$SRC" --python "$HERE/render_vggt.py" -- "$OUT" "$STEP" 2>&1 \
  | tee "$OUT/render.log" | grep --line-buffered -E '^(GPU|PROGRESS)|Error|error:'
[ "$(python3 -c "import json;print(json.load(open('$OUT/render_status.json'))['state'])")" = rendered ]

# 抽帧后按 24 fps 播放（约原速 10 倍的快速预览）；QQ 用 yuv420p 保证兼容
ffmpeg -y -loglevel error -framerate 24 -pattern_type glob -i "$OUT/frames/*.png" \
  -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -movflags +faststart "$OUT/$NAME.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames -of csv=p=0 "$OUT/$NAME.mp4"

# RGB 与深度分开打包，只需图片时不必下载 EXR
( cd "$OUT"
  rm -f "${NAME}_rgb.zip" "${NAME}_depth.zip"
  zip -q -1 -r "${NAME}_rgb.zip" frames cameras.json render_status.json
  zip -q -1 -r "${NAME}_depth.zip" depth cameras.json
  unzip -tq "${NAME}_rgb.zip" && unzip -tq "${NAME}_depth.zip" )
ls -lh "$OUT"/*.zip "$OUT/$NAME.mp4"

baiduyun mkdir "$REMOTE" || true
baiduyun upload "$OUT/${NAME}_rgb.zip" "$OUT/${NAME}_depth.zip" "$OUT/$NAME.mp4" "$REMOTE"
baiduyun ls "$REMOTE"

/usr/bin/python3 -I "$HERE/qq_send_files.py" \
  "【渲染】apartment VGGT 抽帧完成（Cam_MirrorTour，每 ${STEP} 帧取 1 帧，1600×1000）。图片包已上传：$REMOTE" \
  "$OUT/$NAME.mp4"
echo "全部完成"
