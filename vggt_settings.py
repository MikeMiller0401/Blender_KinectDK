"""把 apartment.blend 的渲染设置改为 VGGT 数据集用（基础推荐 + 数据集专用），并保存。"""
import bpy

s = bpy.context.scene
r, c, v = s.render, s.cycles, s.view_settings
s.camera = bpy.data.objects['Cam_MirrorTour']
r.resolution_percentage = 100
c.samples = 512
c.use_adaptive_sampling = True
c.adaptive_threshold = 0.01
c.adaptive_min_samples = 64
c.sample_clamp_indirect = 5.0
c.use_animated_seed = False          # 残余噪声在帧间保持一致
r.use_motion_blur = False             # 模糊会破坏特征匹配
s.camera.data.dof.use_dof = False
v.view_transform = 'AgX'
v.look = 'None'
v.exposure = 0.0
r.image_settings.file_format = 'PNG'
r.image_settings.color_mode = 'RGB'
r.image_settings.color_depth = '16'
for layer in s.view_layers:
    layer.use_pass_z = True
    layer.use_pass_normal = True
bpy.ops.wm.save_mainfile(compress=True)
print('VGGT_SETTINGS', s.camera.name, r.resolution_percentage, c.samples, c.adaptive_threshold,
      c.adaptive_min_samples, c.sample_clamp_indirect, v.look, r.image_settings.color_depth, flush=True)
