"""数据集第二批：关灯相机路径 Cam_LightTour（A 客厅 → B 卧室1 → C 卧室2 → 回到 A）。
相机在 C 内朝里看时关掉 A 的灯，回到 A 时终点位姿与起点完全相同，便于开/关灯对比。
可重复运行（每次重建）。环境变量 SAVE=1 时保存。
关灯设置写在相机自定义属性 light_switch 中，render_dataset.py 默认读取。
"""
import json
import os

import bpy

scene = bpy.context.scene
src = bpy.data.objects['Cam_MirrorTour']
coll = src.users_collection[0]
H = 1.55   # 相机高度，与 MirrorTour 一致
START = (6.8, 4.4)
START_LOOK = (4.6, 1.6, 0.9)

# (帧号, 相机 xy, 注视点 xyz)；帧间距约按行走 0.5 m/s、转向 1~2 s 安排
KEYS = [
    (1, START, START_LOOK),
    (36, START, (4.0, 2.2, 0.9)),
    (60, START, (5.4, 6.2, 1.4)),
    (90, (6.2, 5.2), (5.28, 7.0, 1.4)),
    (140, (5.4, 5.85), (5.28, 8.0, 1.2)),
    (175, (5.28, 6.45), (5.4, 8.5, 1.0)),        # 卧室1 门口
    # 门扇开向室内、沿 x=5.71 伸到 y=7.28，需先直行越过门扇末端再横移
    (215, (5.28, 7.7), (6.0, 9.5, 0.7)),
    (265, (6.1, 7.7), (6.5, 9.6, 0.6)),         # 看床
    (320, (6.4, 7.6), (8.2, 7.3, 1.3)),         # 转向衣柜
    (370, (5.6, 7.7), (5.28, 5.5, 1.4)),        # 转身出门
    (405, (5.28, 7.2), (5.28, 4.5, 1.4)),
    (440, (5.28, 6.05), (8.0, 5.8, 1.45)),
    (470, (5.9, 5.8), (9.0, 5.8, 1.45)),
    # 先在走廊里对正卧室2 门洞再进门，避免斜对门框墙
    (540, (8.3, 5.7), (9.6, 6.4, 1.4)),
    (580, (9.02, 5.65), (9.02, 8.5, 1.2)),
    # 卧室2：门扇（x=8.66）与床尾（x=9.29）之间仅 0.63 m，滑轮直径 0.46 m，相机线取 x=9.02
    # 与门扇并行时保持朝北，越过门扇末端（y=7.28）后再转向床，避免滑轮后移碰门扇
    (620, (9.02, 6.3), (9.2, 8.8, 1.1)),        # 卧室2 门口
    (650, (9.02, 6.9), (9.4, 9.0, 1.0)),
    (690, (9.02, 7.55), (9.8, 9.2, 0.9)),
    (730, (9.02, 7.6), (11.2, 7.7, 0.6)),       # 看床
    (780, (9.02, 7.6), (10.0, 10.0, 1.0)),      # 转向绿植/窗帘，背对门口；此时关客厅灯
    (830, (9.02, 7.55), (11.8, 9.5, 1.2)),
    (870, (9.02, 7.35), (9.0, 5.0, 1.4)),       # 转身出门
    (915, (9.02, 6.4), (9.02, 4.5, 1.4)),
    # 回到走廊后再转向客厅，避免在门洞里贴墙转身
    (950, (8.95, 5.75), (7.2, 4.6, 1.3)),
    (1030, (7.2, 5.6), (5.0, 3.5, 1.1)),
    (1080, START, START_LOOK),                  # 回到起点
    (1116, START, START_LOOK),
]
LIGHT_SWITCH = {'rooms': ['Living'], 'frame': 790, 'fade_frames': 0}

for name in ('Cam_LightTour', 'Cam_LightTour_Target'):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

target = bpy.data.objects.new('Cam_LightTour_Target', None)
target.empty_display_size = 0.1
coll.objects.link(target)
cam = bpy.data.objects.new('Cam_LightTour', src.data.copy())   # 同镜头、同双目参数
cam.data.name = 'Cam_LightTour'
coll.objects.link(cam)
track = cam.constraints.new('TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

for f, (x, y), look in KEYS:
    cam.location = (x, y, H)
    cam.keyframe_insert('location', frame=f)
    target.location = look
    target.keyframe_insert('location', frame=f)
for obj in (cam, target):
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'   # 平滑且不过冲

cam['tour_range'] = [KEYS[0][0], KEYS[-1][0]]
cam['light_switch'] = json.dumps(LIGHT_SWITCH)
print('LIGHTTOUR', cam.name, 'frames', cam['tour_range'], 'switch', LIGHT_SWITCH, flush=True)
if os.environ.get('SAVE') == '1':
    bpy.ops.wm.save_mainfile(compress=True)
    print('SAVED', bpy.data.filepath, flush=True)
