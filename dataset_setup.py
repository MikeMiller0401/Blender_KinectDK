"""数据集第一批：给 apartment.blend 写入灯光分组、实例/语义标注、渲染通道与双目设置。
可重复运行（每次重新分配）。环境变量 SAVE=1 时保存，BASELINE 可改双目基线（米，默认 0.12）。
- 灯光分组：按物体包围盒中心落在哪个房间的地板范围内，写入 Cycles Light Group；世界光为 Sky
- 实例 ID：写入 pass_index（Object Index 通道），同一家具的同类部件共用一个 ID
- 语义类别：写入物体自定义属性 semantic_class（自定义 30 类，见 CLASSES）与 scannet200；
  NYU40 / ScanNet20 经 scannet_labels/ 下的官方映射表由 ScanNet200 派生，均记入 dataset_labels.json
"""
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from mathutils import Vector

scene = bpy.context.scene
layer = scene.view_layers[0]
BASELINE = float(os.environ.get('BASELINE', '0.12'))

# ---- 房间（地板 XY 范围；浴室没有单独地板物体，取剩余的西北角）----
ROOMS = {
    'Living': (0.0, 8.4, 0.0, 5.2),
    'Kitchen': (8.4, 12.6, 0.0, 5.2),
    'Hallway': (0.0, 12.6, 5.2, 6.4),
    'Bathroom': (0.0, 4.66, 6.4, 10.88),
    'Bedroom1': (4.66, 8.4, 6.4, 10.88),
    'Bedroom2': (8.4, 12.6, 6.4, 10.88),
}

CLASSES = ['void', 'wall', 'floor', 'ceiling', 'door', 'window', 'cabinet', 'bed', 'chair', 'sofa',
           'table', 'desk', 'shelf', 'counter', 'books', 'picture', 'mirror', 'curtain', 'rug', 'lamp',
           'plant', 'tv_monitor', 'appliance', 'sink', 'toilet', 'bathtub', 'shower', 'towel', 'pillow',
           'object']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
# 每个物体单独成实例的小物件类别
PER_OBJECT = {'books', 'object', 'pillow', 'towel'}


def root(o):
    while o.parent:
        o = o.parent
    return o


def center(o):
    cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return sum(cs, Vector()) / 8, cs


def room_of(o):
    c, _ = center(o)
    for name, (x0, x1, y0, y1) in ROOMS.items():
        if x0 <= c.x < x1 and y0 <= c.y < y1:
            return name
    return None


# ---- 语义分类 ----
# 浴室是导入模型，部件名无意义，按名称精确指定
BATH = {
    'Box500239': 'floor', 'Box500246': 'floor', 'Object1908874408': 'floor', 'Object1908875173': 'floor',
    'Box500241': 'wall', 'Box500242': 'wall', 'Box500242.001': 'wall', 'Object1908875162': 'wall',
    'Object1908875162.001': 'wall', 'Box500257': 'wall', 'Object1908875172': 'wall', 'W-wall': 'wall',
    'Wall': 'wall', 'Box500247': 'wall', 'Box500249': 'wall', 'Bathroom_DoorCut': 'wall',
    'Line2062826521': 'wall', 'Line2062826522': 'wall',
    'Box500243': 'ceiling', 'Object1908875171': 'ceiling', 'Object1908875174': 'ceiling', 'Box500248': 'ceiling',
    'Box500244': 'shower', 'Object1908875176': 'shower', 'Box500262': 'bathtub', 'toilet': 'toilet',
    'Cylinder8014': 'toilet', 'sink': 'sink', 'Cylinder005': 'mirror', 'FixedWindow001': 'window',
    'Object1908875120': 'curtain', 'Box498807': 'cabinet', 'Box498809': 'cabinet', 'Box2791': 'cabinet',
    'Object1908875175': 'counter', 'Box498851': 'counter', 'Object002': 'table',
    'Object19088z017': 'lamp', 'Box007': 'lamp', 'Object705460063': 'lamp',
    '008': 'towel', '014': 'towel', 'Pattern2D_474667': 'towel', 'Box003': 'rug',
    'Plant GG009': 'plant', 'Object5748': 'plant', 'flower 035': 'plant',
}
# 通用规则：按（物体名 + 根物体名 + 材质名）匹配，先到先得
RULES = [
    (r'mirror|_glass$', 'mirror'),
    (r'^book|magazine', 'books'),
    (r'curtain|finial', 'curtain'),
    (r'^rug|runner', 'rug'),
    (r'^art_|_art', 'picture'),
    (r'^floor_|threshold', 'floor'),
    # 门窗要先于墙判断：部分窗框、门套以 Wall_ 开头
    (r'door|casing|lining|lever|kick', 'door'),
    (r'window|win\d|sill', 'window'),
    (r'^wall_|backsplash|skirting|slab', 'wall'),
    (r'^ceiling$|^soffit', 'ceiling'),
    (r'downlight|pendant|ceil_|cove|lamp|_led_|bulb', 'lamp'),
    (r'leaves|plant|herb|pot$|soil|pebbles', 'plant'),
    (r'monitor|tv screen', 'tv_monitor'),
    (r'fridge|hood|cooktop|kettle', 'appliance'),
    (r'^sink|faucet', 'sink'),
    (r'counter_[es]$|counter_s_sinkcut|island_top', 'counter'),
    (r'stool|chair', 'chair'),
    (r'sofa', 'sofa'),
    (r'^bed\.0(0[4-7]|1[89])$', 'pillow'),
    (r'bed|mattress', 'bed'),
    (r'desk', 'desk'),
    (r'diningtable|nightstand_.*(leg|body|front|knob)', None),  # 下面按根物体再判
    (r'shelf|cube', 'shelf'),
    (r'wardrobe|kitchen_(base|upper|plinth)|pantry|shoe_cabinet|island_(body|plinth)|nightstand', 'cabinet'),
    (r'keyboard|bowl|fruit|vase|jar|plate|pitcher|cutting|keybowl|body|base|buttons|feet|keys|leds|wheel',
     'object'),
]


def classify(o):
    r = root(o).name
    if r == 'Bathroom modern':
        return BATH.get(o.name, 'object')
    name = o.name.lower()
    if r == 'DiningTable':
        return 'table' if name.startswith('diningtable') else 'object'
    if r.startswith('Nightstand'):
        return 'lamp' if name.startswith('lamp') else 'cabinet'
    if r in ('King Bed', 'Cozy bed'):
        return 'pillow' if re.match(r'^bed\.0(0[4-7]|1[89])$', name) else 'bed'
    for pat, cls in RULES:
        if cls and re.search(pat, name):
            return cls
    return 'object'


# ---- ScanNet200 标注，并经官方映射表派生 NYU40 / ScanNet20 ----
LABEL_DIR = Path(__file__).resolve().parent / 'scannet_labels'


def load_scannet():
    tree = ast.parse((LABEL_DIR / 'scannet200_constants.py').read_text())
    const = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ('VALID_CLASS_IDS_200', 'CLASS_LABELS_200', 'VALID_CLASS_IDS_20',
                                     'CLASS_LABELS_20')}
    with open(LABEL_DIR / 'scannetv2-labels.combined.tsv', newline='', encoding='utf-8') as f:
        rows = {int(r['id']): r for r in csv.DictReader(f, delimiter='\t') if r['id'].strip()}
    return const, rows


SN, SN_ROWS = load_scannet()
S200_ID = dict(zip(SN['CLASS_LABELS_200'], SN['VALID_CLASS_IDS_200']))
NYU40_NAMES = {int(r['nyu40id']): r['nyu40class'] for r in SN_ROWS.values() if r['nyu40id'].strip()}

# 浴室导入模型：部件名无意义，按名称精确指定；未列出的再按材质判断
BATH200 = {
    'Box500239': 'floor', 'Object1908874408': 'floor', 'Object1908875173': 'floor', 'Box500246': 'shower floor',
    'Box500241': 'wall', 'Box500242': 'wall', 'Box500242.001': 'wall', 'Object1908875162': 'wall',
    'Object1908875162.001': 'wall', 'Box500257': 'wall', 'Object1908875172': 'wall', 'W-wall': 'wall',
    'Wall': 'wall', 'Bathroom_DoorCut': 'wall', 'Line2062826521': 'wall', 'Line2062826522': 'wall',
    'Box500247': 'shower wall', 'Box500249': 'shower wall',
    'Box500243': 'ceiling', 'Object1908875171': 'ceiling', 'Object1908875174': 'ceiling', 'Box500248': 'ceiling',
    'Box500244': 'shower door', 'Object1908875176': 'shower door', 'Box500262': 'bathtub',
    'toilet': 'toilet', 'Cylinder8014': 'toilet', 'sink': 'sink', 'Cylinder005': 'mirror',
    'FixedWindow001': 'window', 'Object1908875120': 'blinds',
    'Box498807': 'bathroom cabinet', 'Box498809': 'bathroom cabinet', 'Box2791': 'bathroom vanity',
    'Object1908875175': 'bathroom counter', 'Box498851': 'bathroom counter', 'Object002': 'end table',
    'Object19088z017': 'ceiling light', 'Box007': 'light', 'Object705460063': 'light',
    '008': 'towel', '014': 'towel', 'Pattern2D_474667': 'towel', 'Box003': 'mat',
    'Mesh318': 'rack', 'Rectangle091': 'rack', 'Rectangle091.001': 'rack',
    'Plant GG009': 'plant', 'Object5748': 'plant', 'flower 035': 'plant',
    'desirefx.me_011': 'decoration', 'Line2062826519': 'decoration',
}
BATH200_BY_MATERIAL = [
    (r'material #16|^l1$', 'plant'),
    (r'^cera$|^wax$|torch', 'candle'),
    (r'bathaccessories|glass labeled|rituelle|cream glass|brown glass|ior glass|green glass', 'bottle'),
    (r'su,cream|^dec\d', 'decoration'),
]


def scannet200(o):
    r = root(o).name.lower()
    n = o.name.lower()
    if r == 'bathroom modern':
        if o.name in BATH200:
            return BATH200[o.name]
        mats = ','.join(m.name.lower() for m in o.data.materials if m)
        for pat, cls in BATH200_BY_MATERIAL:
            if re.search(pat, mats):
                return cls
        return 'object'
    if r in ('king bed', 'cozy bed'):
        if re.match(r'^bed\.0(0[4-7]|1[89])$', n):
            return 'pillow'
        return 'mattress' if n in ('mattress', 'bed.014') else 'bed'
    if r.startswith('nightstand'):
        return 'lamp' if n.startswith('lamp') else 'nightstand'
    if r == 'diningtable':
        return 'dining table' if n.startswith('diningtable') else ('bowl' if n == 'dining_bowl' else 'object')
    if r == 'shelfs':
        if n.startswith('sm_tv'):
            return 'tv'
        if re.match(r'^(book|magazine)', n):
            return 'book'
        return 'tv stand' if re.match(r'^cube(\.00[0-4])?$', n) else 'shelf'
    if r.startswith('wireless dell keyboard'):
        return 'keyboard' if n in ('body', 'keys', 'feet', 'leds') else 'mouse'
    if r == 'desk':
        return 'lamp' if n.startswith('desklamp') else ('monitor' if n.startswith('monitor') else 'desk')
    if r == 'openshelves':
        if n.startswith('kitchen_shelf'):
            return 'shelf'
        return 'container' if n.startswith('jar') else ('plate' if n.startswith('plate') else 'water pitcher')
    if r == 'island':
        if n == 'island_top':
            return 'kitchen counter'
        return 'bowl' if n == 'island_bowl' else ('object' if 'fruit' in n else 'kitchen cabinet')
    if r in ('counter_east', 'counter_south'):
        for pat, cls in [(r'^counter_', 'kitchen counter'), (r'^cooktop', 'stove'), (r'^sink|^faucet', 'sink'),
                         (r'^kettle', 'coffee kettle'), (r'^kitchen_herb', 'potted plant'), (r'^cutting', 'object')]:
            if re.search(pat, n):
                return cls
        return 'kitchen cabinet'
    if r.startswith('wallcabinet'):
        return 'light' if 'led' in n else 'kitchen cabinet'
    if r == 'shoecabinet':
        return 'bowl' if 'keybowl' in n else ('decoration' if 'vase' in n else 'cabinet')
    if r.startswith('wardrobe'):
        return 'mirror' if 'mirror' in n else 'wardrobe'
    if r == 'wallshelves_bed2':
        return 'book' if re.match(r'^(book|magazine)', n) else 'shelf'
    for pat, cls in [
        (r'^(pantry)$', 'kitchen cabinet'), (r'^fridge$', 'refrigerator'), (r'^hood$', 'range hood'),
        (r'^(indoor plant|potted plant 01)$', 'potted plant'), (r'^office chair$', 'office chair'),
        (r'^leather sofa$', 'couch'), (r'^chair\d$', 'chair'), (r'^stool\d$', 'stool'),
        (r'^curtains', 'curtain'), (r'^(rug_|runner)', 'mat'), (r'^art_', 'picture'), (r'mirror', 'mirror'),
        (r'^(downlight|ceil_|pendant)', 'ceiling light'), (r'^cove', 'light'), (r'^floorlamp', 'lamp'),
        (r'^(ceiling|soffit)', 'ceiling'),
    ]:
        if re.search(pat, r):
            return cls
    # 结构与门窗按部件名判断：门套/窗台/窗框可能挂在 Wall_ 名下
    for pat, cls in [(r'casing|lining', 'doorframe'), (r'sill', 'windowsill'), (r'window|win\d', 'window'),
                     (r'door|lever|kick', 'door'), (r'^floor_|threshold|slab', 'floor'),
                     (r'^wall_|backsplash|skirting', 'wall')]:
        if re.search(pat, n):
            return cls
    return 'object'


# ScanNet200 中每个物体单独成实例的小物件
PER_OBJECT_200 = {'book', 'bottle', 'plate', 'bowl', 'object', 'candle', 'decoration', 'pillow', 'towel',
                  'container'}


# ---- 1. 灯光分组 ----
for lg in list(layer.lightgroups):
    layer.lightgroups.remove(lg)
for name in list(ROOMS) + ['Sky', 'Misc']:
    layer.lightgroups.add(name=name)
scene.world.lightgroup = 'Sky'
light_rooms = defaultdict(list)
for o in scene.objects:
    if o.type not in {'LIGHT', 'MESH'}:
        continue
    if o.users_collection[0].name == 'Cameras':
        o.lightgroup = 'Misc'
        continue
    room = room_of(o) or 'Misc'
    o.lightgroup = room
    o['room'] = room
    if o.type == 'LIGHT':
        light_rooms[room].append(o.name)

# ---- 2. 语义与实例 ----
instances = {}
labels = {}
counts = Counter()
counts200 = Counter()
for o in sorted(scene.objects, key=lambda o: o.name):
    if o.type != 'MESH':
        continue
    if o.users_collection[0].name == 'Cameras':
        o.pass_index = 0   # 机位道具对相机不可见，只在镜中出现
        continue
    cls = classify(o)
    s200 = scannet200(o)
    if s200 not in S200_ID:
        raise RuntimeError(f'{o.name}: {s200!r} 不在 ScanNet200 类别表中')
    # 实例按（根物体, ScanNet200 类, 自定义类）划分，保证同一实例在各套类别下都只有一个类
    key = (o.name,) if cls in PER_OBJECT or s200 in PER_OBJECT_200 else (root(o).name, s200, cls)
    if key not in instances:
        instances[key] = len(instances) + 1
    o.pass_index = instances[key]
    o['semantic_class'] = cls
    o['scannet200'] = s200
    sid = S200_ID[s200]
    nyu40 = int(SN_ROWS[sid]['nyu40id'])
    labels.setdefault(o.pass_index, {
        'class': cls, 'class_id': CLASS_ID[cls], 'scannet200': s200, 'scannet200_id': sid,
        'nyu40_id': nyu40, 'nyu40_class': NYU40_NAMES[nyu40],
        'scannet20_id': SN['VALID_CLASS_IDS_20'].index(nyu40) if nyu40 in SN['VALID_CLASS_IDS_20'] else 255,
        'room': o.get('room'), 'objects': []})['objects'].append(o.name)
    counts[cls] += 1
    counts200[s200] += 1
txt = bpy.data.texts.get('dataset_labels.json') or bpy.data.texts.new('dataset_labels.json')
txt.from_string(json.dumps({
    'classes': CLASSES,
    'schemes': {
        'scannet200': {'note': '语义图值为 ScanNet 原始类别 ID（16-bit PNG），0 = 未标注',
                       'ids': list(SN['VALID_CLASS_IDS_200']), 'names': list(SN['CLASS_LABELS_200'])},
        'nyu40': {'note': '语义图值为 NYU40 ID（1-40），0 = 未标注；由 scannetv2-labels.combined.tsv 映射',
                  'names': {str(k): v for k, v in sorted(NYU40_NAMES.items())}},
        'scannet20': {'note': '语义图值为 0-19 训练索引，255 = 忽略/未标注',
                      'nyu40_ids': list(SN['VALID_CLASS_IDS_20']), 'names': list(SN['CLASS_LABELS_20'])},
    },
    'instances': labels}, ensure_ascii=False, indent=1))

# ---- 3. 渲染通道 ----
for p in ('use_pass_z', 'use_pass_normal', 'use_pass_diffuse_color', 'use_pass_diffuse_direct',
          'use_pass_diffuse_indirect', 'use_pass_glossy_color', 'use_pass_glossy_direct',
          'use_pass_glossy_indirect', 'use_pass_transmission_color', 'use_pass_transmission_direct',
          'use_pass_transmission_indirect', 'use_pass_emit', 'use_pass_environment', 'use_pass_object_index'):
    setattr(layer, p, True)
layer.cycles.denoising_store_passes = True   # 供合成器给 shading 降噪
# 半透明玻璃（如浴室窗 alpha≈0.19）在默认阈值 0.5 下会逐采样随机穿透，深度/实例出现噪点；
# 降到 0.1 后玻璃稳定记为表面，完全透明处（alpha=0，如叶片镂空）仍会穿过
layer.pass_alpha_threshold = 0.1

# ---- 4. 双目：左眼即原相机位姿，右眼沿相机 +X 平移基线，平行光轴 ----
scene.render.use_multiview = True
scene.render.views_format = 'STEREO_3D'
scene.render.image_settings.views_format = 'INDIVIDUAL'
for cam in (o for o in bpy.data.objects if o.type == 'CAMERA'):
    st = cam.data.stereo
    st.convergence_mode = 'PARALLEL'
    st.interocular_distance = BASELINE
    st.pivot = 'LEFT'

print('LIGHTS', {k: len(v) for k, v in light_rooms.items()}, flush=True)
print('LIGHTS_DETAIL', json.dumps(light_rooms, ensure_ascii=False), flush=True)
print('CLASSES', dict(counts.most_common()), flush=True)
print('INSTANCES', len(instances), flush=True)
print('SCANNET200', dict(counts200.most_common()), flush=True)
print('S200_FALLBACK', sorted(o.name for o in scene.objects if o.get('scannet200') == 'object'), flush=True)
print('OBJECT_FALLBACK', [o.name for o in scene.objects if o.get('semantic_class') == 'object'
                          and root(o).name != 'Bathroom modern'][:80], flush=True)
if os.environ.get('SAVE') == '1':
    bpy.ops.wm.save_mainfile(compress=True)
    print('SAVED', bpy.data.filepath, flush=True)
