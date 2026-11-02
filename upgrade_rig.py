"""细化 Cam_MirrorTour 机位道具（镜中可见的相机+三脚架+滑轮）。
坐标均为 Cam_MirrorTour_Rig 局部系：镜头朝 -Z，Y 向上，机身位于 z≈0.10~0.16。
环境变量 SAVE=1 时保存回当前文件，否则只在内存中修改（供预览）。"""
import math
import os

import bmesh
import bpy
from mathutils import Vector

RIG = bpy.data.objects['Cam_MirrorTour_Rig']
COLL = bpy.data.objects['CamRig_Body'].users_collection[0]
PREFIXES = ('CamRig_', 'Dolly_', 'Tripod_', 'Head_', 'Cam_MirrorTour_Column')
if 'CamRig_Strap' in bpy.data.objects:
    raise SystemExit('已细化过，跳过')


def principled(mat):
    return next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree, principled(m)


def set_inputs(bsdf, **kw):
    for k, v in kw.items():
        bsdf.inputs[k.replace('_', ' ')].default_value = v


def add_obj(name, mesh, mat):
    o = bpy.data.objects.new(name, mesh)
    COLL.objects.link(o)
    o.parent = RIG
    o.visible_camera = False
    mesh.materials.append(mat)
    return o


def cylinder(name, mat, x, y, z0, r, depth, seg=32):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=depth)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(x, y, z0 + depth / 2))
    bm.to_mesh(me)
    bm.free()
    o = add_obj(name, me, mat)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(40))
    return o


# ---- 1. 着色：低段数管件改平滑，倒角改硬化法线，避免棱面和大平面上的渐变斑 ----
for o in bpy.data.objects:
    if o.type != 'MESH' or not o.name.startswith(PREFIXES):
        continue
    if not o.data.polygons[0].use_smooth:
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(35))
    for m in o.modifiers:
        if m.type == 'BEVEL':
            m.harden_normals = True
            m.segments = max(m.segments, 4)

# ---- 2. 材质 ----
# 铝管：拉丝金属，降低亮度，避免看起来像乳白玻璃
set_inputs(principled(bpy.data.materials['CamRig_Silver']),
           Base_Color=(0.56, 0.57, 0.59, 1), Roughness=0.3, Anisotropic=0.6)
set_inputs(principled(bpy.data.materials['CamRig_Metal']), Roughness=0.42)

# 机身蒙皮：细颗粒皮纹
leather, nt, b = new_mat('CamRig_Leather')
tc = nt.nodes.new('ShaderNodeTexCoord')
vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.feature = 'SMOOTH_F1'
vor.inputs['Scale'].default_value = 1400
nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
bump = nt.nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = 0.35; bump.inputs['Distance'].default_value = 0.0004
nt.links.new(vor.outputs['Distance'], bump.inputs['Height'])
nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
rr = nt.nodes.new('ShaderNodeMapRange')
rr.inputs['To Min'].default_value = 0.42; rr.inputs['To Max'].default_value = 0.62
nt.links.new(vor.outputs['Distance'], rr.inputs['Value'])
nt.links.new(rr.outputs['Result'], b.inputs['Roughness'])
set_inputs(b, Base_Color=(0.014, 0.014, 0.015, 1), Coat_Weight=0.05, Coat_Roughness=0.4)
for n in ('CamRig_Body', 'CamRig_Grip', 'CamRig_GripFace'):
    bpy.data.objects[n].data.materials[0] = leather

# 对焦/变焦环：沿圆周的防滑纹
ribs, nt, b = new_mat('CamRig_RingRubber')
tc = nt.nodes.new('ShaderNodeTexCoord')
grad = nt.nodes.new('ShaderNodeTexGradient'); grad.gradient_type = 'RADIAL'
nt.links.new(tc.outputs['Object'], grad.inputs['Vector'])
mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
mul.inputs[1].default_value = 2 * math.pi * 120
nt.links.new(grad.outputs['Fac'], mul.inputs[0])
sine = nt.nodes.new('ShaderNodeMath'); sine.operation = 'SINE'
nt.links.new(mul.outputs[0], sine.inputs[0])
bump = nt.nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = 0.8; bump.inputs['Distance'].default_value = 0.0006
nt.links.new(sine.outputs[0], bump.inputs['Height'])
nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
set_inputs(b, Base_Color=(0.012, 0.012, 0.013, 1), Roughness=0.62)
for n in ('CamRig_FocusRing', 'CamRig_ZoomRing'):
    bpy.data.objects[n].data.materials[0] = ribs

# 镜片：深色高光 + 薄膜镀膜（绿/紫反光）
set_inputs(principled(bpy.data.materials['CamRig_LensGlass']),
           Base_Color=(0.003, 0.004, 0.006, 1), Roughness=0.0, Coat_Weight=0.0,
           Thin_Film_Thickness=420.0, Thin_Film_IOR=1.38, Specular_IOR_Level=0.7)

# 屏幕：微亮的取景画面
scr = bpy.data.materials['CamRig_Screen']; nt = scr.node_tree; b = principled(scr)
tc = nt.nodes.new('ShaderNodeTexCoord')
# 上墙下地板的简化取景画面
sep = nt.nodes.new('ShaderNodeSeparateXYZ')
nt.links.new(tc.outputs['Generated'], sep.inputs['Vector'])
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.30, 0.17, 0.08, 1)
ramp.color_ramp.elements[1].color = (0.80, 0.77, 0.72, 1)
ramp.color_ramp.elements[0].position = 0.38; ramp.color_ramp.elements[1].position = 0.45
nt.links.new(sep.outputs['Y'], ramp.inputs['Fac'])
nt.links.new(ramp.outputs['Color'], b.inputs['Emission Color'])
set_inputs(b, Emission_Strength=0.6, Roughness=0.04)

strap_mat, nt, b = new_mat('CamRig_Strap')
tc = nt.nodes.new('ShaderNodeTexCoord')
wave = nt.nodes.new('ShaderNodeTexWave'); wave.inputs['Scale'].default_value = 900
nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
bump = nt.nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = 0.5; bump.inputs['Distance'].default_value = 0.0003
nt.links.new(wave.outputs['Fac'], bump.inputs['Height'])
nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
set_inputs(b, Base_Color=(0.008, 0.008, 0.009, 1), Roughness=0.9, Specular_IOR_Level=0.3)

body_plastic = bpy.data.materials['CamRig_Body']
metal = bpy.data.materials['CamRig_Metal']
rubber = bpy.data.materials['CamRig_Rubber']

# ---- 3. 几何 ----
# 取景器改成五棱镜外形：顶部收窄、前沿后倾
fin = bpy.data.objects['CamRig_Finder'].data
for v in fin.vertices:
    if v.co.y > 0.06:
        v.co.x *= 0.5
        if v.co.z < 0.13:
            v.co.z += 0.012

# 前镜片改成凸面
glass = bpy.data.objects['CamRig_FrontGlass']
R, K, SEG, zb, zf, h = 0.0301, 8, 64, 0.0065, 0.0045, 0.0032
bm = bmesh.new()
rings = []
center = bm.verts.new((0, 0, zf - h))
for k in range(1, K + 1):
    r = R * k / K
    z = zf - h * (1 - (r / R) ** 2)
    rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / SEG), r * math.sin(2 * math.pi * i / SEG), z))
                  for i in range(SEG)])
back = [bm.verts.new((v.co.x, v.co.y, zb)) for v in rings[-1]]
for i in range(SEG):
    j = (i + 1) % SEG
    bm.faces.new((center, rings[0][j], rings[0][i]))
    for a, c in zip(rings, rings[1:]):
        bm.faces.new((a[i], a[j], c[j], c[i]))
    bm.faces.new((rings[-1][i], rings[-1][j], back[j], back[i]))
bm.faces.new(back)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
gm = bpy.data.meshes.new('CamRig_FrontGlass_Dome')
bm.to_mesh(gm); bm.free()
gm.materials.append(bpy.data.materials['CamRig_LensGlass'])
glass.data = gm
glass.modifiers.clear()
gm.shade_smooth()

# 遮光罩：外扩的薄壁筒
bm = bmesh.new()
r0, r1, L, t = 0.0392, 0.047, 0.026, 0.0012
outer0, outer1, inner0, inner1 = [], [], [], []
for i in range(SEG):
    c, s = math.cos(2 * math.pi * i / SEG), math.sin(2 * math.pi * i / SEG)
    outer0.append(bm.verts.new((r0 * c, r0 * s, 0.002)))
    outer1.append(bm.verts.new((r1 * c, r1 * s, -L)))
    inner0.append(bm.verts.new(((r0 - t) * c, (r0 - t) * s, 0.002)))
    inner1.append(bm.verts.new(((r1 - t) * c, (r1 - t) * s, -L)))
for i in range(SEG):
    j = (i + 1) % SEG
    bm.faces.new((outer0[i], outer0[j], outer1[j], outer1[i]))
    bm.faces.new((inner0[j], inner0[i], inner1[i], inner1[j]))
    bm.faces.new((outer1[i], outer1[j], inner1[j], inner1[i]))
    bm.faces.new((inner0[i], inner0[j], outer0[j], outer0[i]))
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
hm = bpy.data.meshes.new('CamRig_Hood')
bm.to_mesh(hm); bm.free()
add_obj('CamRig_Hood', hm, body_plastic)
hm.shade_smooth(); hm.set_sharp_from_angle(angle=math.radians(40))

# 机背按键与后拨盘（屏幕右侧及左上）
BACK = 0.157
cylinder('CamRig_RearDial', metal, 0.049, -0.012, BACK, 0.0105, 0.0018, 48)
cylinder('CamRig_BtnOK', metal, 0.049, -0.012, BACK, 0.0042, 0.0030)
for n, (x, y) in enumerate([(0.049, 0.010), (0.049, 0.024), (0.049, -0.034), (-0.054, 0.031), (-0.042, 0.031)]):
    cylinder(f'CamRig_Btn{n}', metal, x, y, BACK, 0.0034, 0.0016)
thumb = cylinder('CamRig_ThumbRest', rubber, 0.052, 0.032, BACK, 0.0075, 0.0022)
thumb.scale = (1.0, 0.7, 1.0)

# 背带：从两侧吊耳垂下，在镜头下方形成 U 形；宽度沿 Z 方向
ctrl = [(-0.066, 0.028, 0.120), (-0.074, 0.010, 0.112), (-0.078, -0.060, 0.098), (-0.072, -0.140, 0.082),
        (-0.045, -0.200, 0.068), (0.0, -0.222, 0.062)]
ctrl = ctrl + [(-x, y, z) for x, y, z in reversed(ctrl[:-1])]
P = [Vector(p) for p in ctrl]


def catmull(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


pts = []
ext = [P[0]] + P + [P[-1]]
for i in range(1, len(ext) - 2):
    for s in range(12):
        pts.append(catmull(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], s / 12))
pts.append(P[-1])
bm = bmesh.new()
W = 0.0125
pairs = []
for i, p in enumerate(pts):
    # 吊耳附近收窄成穿环的细带
    end = min(i, len(pts) - 1 - i)
    w = W * (0.35 if end < 4 else 1.0)
    pairs.append((bm.verts.new(p + Vector((0, 0, -w))), bm.verts.new(p + Vector((0, 0, w)))))
for (a0, a1), (b0, b1) in zip(pairs, pairs[1:]):
    bm.faces.new((a0, b0, b1, a1))
sm = bpy.data.meshes.new('CamRig_Strap')
bm.to_mesh(sm); bm.free()
strap = add_obj('CamRig_Strap', sm, strap_mat)
sol = strap.modifiers.new('厚度', 'SOLIDIFY'); sol.thickness = 0.0016; sol.offset = 0
sm.shade_smooth()

print('RIG_UPGRADED', flush=True)
if os.environ.get('SAVE') == '1':
    bpy.ops.wm.save_mainfile(compress=True)
    print('SAVED', bpy.data.filepath, flush=True)
