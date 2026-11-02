#!/usr/bin/env bash
# 抽帧渲染 apartment → 合成视频 → 打包 → 上传百度网盘 → QQ 通知
# 用法：[CAMERA=相机名] render_sampled.sh [输出目录]   重跑同一目录会跳过已渲染的帧
#       不设 CAMERA 时用文件里的活动相机
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../apartment.blend"
STEP=10
CAMERA=${CAMERA:-}
TAG=${CAMERA:+_${CAMERA#Cam_}}
OUT=${1:-"$HERE/gpu_100pct_every${STEP}${TAG}_$(date -u +%Y%m%dT%H%M%SZ)"}
REMOTE=/UbuntuData/Renders/apartment
BLENDER=/home/tcy/Applications/Blender4.5.12/blender
# SUFFIX 用于区分同一场景的多个版本；网盘同名文件会被跳过上传
NAME=apartment${TAG}_100pct_every${STEP}${SUFFIX:-}
mkdir -p "$OUT"
echo "输出目录：$OUT"

"$BLENDER" -b "$SRC" --python "$HERE/render_sampled.py" -- "$OUT" "$STEP" "$CAMERA" 2>&1 \
  | tee "$OUT/render.log" | grep --line-buffered -E '^(GPU|PROGRESS)|Error|error:'
[ "$(python3 -c "import json;print(json.load(open('$OUT/render_status.json'))['state'])")" = rendered ]

# 抽帧后按 24 fps 播放（约原速 10 倍的快速预览）
ffmpeg -y -loglevel error -framerate 24 -pattern_type glob -i "$OUT/frames/*.png" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT/$NAME.mp4"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames -of csv=p=0 "$OUT/$NAME.mp4"

( cd "$OUT" && rm -f "$NAME.zip" && zip -q -r "$NAME.zip" "$NAME.mp4" frames render_status.json && unzip -tq "$NAME.zip" )
ls -lh "$OUT/$NAME.zip"

baiduyun mkdir "$REMOTE" || true
baiduyun upload "$OUT/$NAME.zip" "$REMOTE"
baiduyun meta "$REMOTE/$NAME.zip"

/usr/bin/python3 -I "$HERE/qq_notify.py" "apartment 抽帧渲染完成并已上传：$REMOTE/$NAME.zip（$(du -h "$OUT/$NAME.zip" | cut -f1)）"
echo "全部完成"
