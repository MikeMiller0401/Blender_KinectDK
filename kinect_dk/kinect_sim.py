"""Azure Kinect DK 模拟第二步：把 render_kinect.py 的针孔渲染变成 DK 深度相机输出（NFOV unbinned 640×576）。
用 anaconda Python（numpy + cv2 + PIL）运行：python3 kinect_sim.py 渲染输出目录 [--seed 0] [--preview]
每个视角写入 <名称>/：
  depth.png           模拟深度，uint16 毫米，0 = 无效（与 SDK 一致）
  ir.png              模拟主动红外图，uint16
  depth_gt.png        同视角干净深度真值，uint16 毫米（只去掉六边形外）
  depth_to_color.png  模拟深度投影到彩色相机 1280×720，uint16 毫米
  invalid_reason.png  uint8：0 有效 1 照明区外 2 信号弱 3 过曝 4 边缘混合像素 6 玻璃/镜子正对发射器的镜面亮斑
模型（参数见 PARAMS，写入 sim_params.json）：
  红外信号 S = k · E(u,v) · I，I 为只开相机处点光源时 Cycles 的辐亮度（换算到 ρ·cos/r² 单位），
  漫反射部分按类别把可见光反照率换成 850 nm 反射率；k 与照度分布 E 由 calibrate_dk.py 用真机墙面标定。
  深度噪声 σ = sqrt(σ0² + σ1² · S_ref / S)，S_ref 为 1 m 处正对的 ρ=0.5 平面；与真机 test 序列的时间噪声核对。
  内凹棱角：真机（DK_recorder 墙角，0.5–0.6 m）测得比几何更近、全部有效，偏差随到棱线的距离指数衰减。
  玻璃：ToF 透过玻璃测到后方；镜子：测到虚像（沿反射展开的总路程），信号按往返路程与镜面反射率估计；
  两者近乎正对时发射器的镜面反射过曝（真机亮斑半径约 20 px）。
  无效判据：低信号概率曲线按真机 test 序列，过曝阈值按 test 与 DK_recorder 录像（约 20000 DN）。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dk_common import undistort  # noqa: E402

PARAMS = {
    'sigma0_mm': 0.5,             # 基础噪声
    'sigma1_mm': 1.0,             # 参考信号下的散粒噪声项（1 m、正对、ρ=0.5）
    # 有效概率 = 1 / (1 + (ir50 / IR)^slope)，IR 为红外信号（DN）；按标定 k 的同一台真机 test 序列拟合：
    # IR 0–10 → 3%，10–20 → 17%，20–40 → 48%，40–60 → 81%，100–150 → 95%（新设备墙面录像低 IR 像素都在视场边缘，不采用）
    'low_signal_ir50': 28.0,
    'low_signal_slope': 2.5,
    'saturation': 20000,          # 真机红外到约 20000 仍有效，过曝像素读数 65535 且无效
    'edge_jump': [0.03, 0.015],   # 四个方向二阶差分 > 2·max(a, b·d) 视为混合像素
    # 断层两侧的无效带：半宽 = a + b·d（像素，d 米）+ 沿边缘缓变的随机量 + 逐像素毛刺；
    # 按真机 test 序列标定（台阶边缘无效段宽度中位数：近处 5–6 px，>3 m 约 11 px）
    'edge_halfwidth_px': [0.3, 0.4],
    'edge_width_noise_px': 1.3,
    'edge_jitter_px': 0.7,
    # 内凹棱角（法线夹角 > corner_angle_deg）：偏差 = corner_bias_mm · exp(-d / corner_decay_m)，d 为沿表面到棱线的距离；
    # 真机分段中位数 0–1 cm −22，1–2 cm −17，2–4 cm −11，4–7 cm −5 mm，7 cm 外约 0
    'corner_bias_mm': -26.0,
    'corner_decay_m': 0.035,
    'corner_angle_deg': 30.0,
    'glass_transmission': 0.85,   # 单程透过率
    'specular_cos': 0.999,        # 玻璃/镜面入射角余弦高于此值时发射器镜面反射过曝（约 2.5°）
    'mirror_reflectance': 0.9,    # 镜面单次反射率（850 nm）
    'mirror_rho': 0.4,            # 镜中物体的红外反射率（未逐物体取值）
    'ir_gain': 2.0, 'ir_read_noise': 10.0,
    'rho_ir_overrides': {'leaf': 0.55, 'textile_min': 0.4},
}
TEXTILE = {'pillow', 'curtain', 'couch', 'bed', 'blanket', 'towel', 'mat', 'armchair', 'clothes', 'sofa chair'}
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def rho_ir_table(labels, max_id):
    """按实例给 850 nm 反射率规则：0 = 用可见光反照率，>0 = 固定值，<0 = 取 max(反照率, -值)。"""
    table = np.zeros(max_id + 1, np.float32)
    for k, inst in labels.get('instances', {}).items():
        i = int(k)
        if i > max_id:
            continue
        if any('leaf' in o.lower() or 'leaves' in o.lower() for o in inst['objects']):
            table[i] = PARAMS['rho_ir_overrides']['leaf']
        elif inst['scannet200'] in TEXTILE:
            table[i] = -PARAMS['rho_ir_overrides']['textile_min']
    return table


def concave_pairs(P, n, valid, z, step):
    """相隔 2·step 像素的两点法线夹角大、深度连续，且互在对方平面之前（内凹）。"""
    H, W = z.shape
    cmax = np.cos(np.radians(PARAMS['corner_angle_deg']))
    pad = lambda a: np.pad(a, ((step, step), (step, step)) + ((0, 0),) * (a.ndim - 2), mode='edge')   # noqa: E731
    Pp, Np, Vp = pad(P), pad(n), pad(valid)
    out = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sl_a = (slice(step - dy * step, step - dy * step + H), slice(step - dx * step, step - dx * step + W))
        sl_b = (slice(step + dy * step, step + dy * step + H), slice(step + dx * step, step + dx * step + W))
        a, b, na, nb = Pp[sl_a], Pp[sl_b], Np[sl_a], Np[sl_b]
        ab = b - a
        out |= (Vp[sl_a] & Vp[sl_b] & ((na * nb).sum(2) < cmax) & (np.abs(ab[..., 2]) < 0.02 + 0.05 * z)
                & ((ab * na).sum(2) > 0) & ((-ab * nb).sum(2) > 0))
    return out


def corner_bias(z, xn, n_cam, ray, cos, valid, f):
    """内凹棱角的深度偏差（米）。棱线取相邻像素即判为内凹的细线，且在更大尺度上仍内凹
    （尺度约 3–6 cm，排除门缝、把手槽等小凹槽；真机只测过房间墙角这类大棱角）。"""
    P = np.dstack([xn * z[..., None], z])
    n = n_cam * -np.sign((n_cam * ray).sum(2, keepdims=True))    # 法线朝向相机
    large = np.zeros(z.shape, bool)
    for s in (4, 8, 16, 32, 64):                               # 按深度选步长，使 s·z/f 落在 3–6 cm
        large |= concave_pairs(P, n, valid, z, s) & (s * z / f >= 0.03) & (s * z / f < 0.06)
    large = cv2.dilate(large.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    crease = concave_pairs(P, n, valid, z, 1) & large
    if not crease.any():
        return np.zeros_like(z)
    dist_px, lab = cv2.distanceTransformWithLabels((~crease).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    z_cr = np.concatenate([[0], z[crease]])[lab]              # 最近棱线像素的深度（标签按行优先顺序编号）
    d_m = dist_px * z / f / np.maximum(cos, 0.3)
    same = np.abs(z - z_cr) < 0.02 + 2 * d_m                  # 同一结构（不跨越前景物体）
    return np.where(valid & same, PARAMS['corner_bias_mm'] / 1000 * np.exp(-d_m / PARAMS['corner_decay_m']), 0)


def simulate(view, rig, cal, labels, rng):
    p = np.load(view / 'ir_passes.npz')
    through = np.load(view / 'through.npz')['depth']
    K, dist = np.array(rig['depth']['K']), np.array(rig['depth']['dist_k1k2p1p2k3k4k5k6'])
    W, H = rig['depth']['size']
    Kr = np.array(rig['pinhole_render']['K'])
    mask, E = cal['mask'], cal['falloff']

    # DK 像素 → 针孔渲染坐标
    yy, xx = np.mgrid[0:H, 0:W]
    xn = undistort(np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float64), K, dist).reshape(H, W, 2)
    mx = (xn[..., 0] * Kr[0, 0] + Kr[0, 2]).astype(np.float32)
    my = (xn[..., 1] * Kr[1, 1] + Kr[1, 2]).astype(np.float32)
    near = lambda a: cv2.remap(a, mx, my, cv2.INTER_NEAREST)   # noqa: E731
    lin = lambda a: cv2.remap(a, mx, my, cv2.INTER_LINEAR)     # noqa: E731

    z = near(p['depth'])
    z_through = near(through)
    idx = near(p['index'].astype(np.float32)).astype(np.int32)
    nrm = near(p['normal'])
    ray = np.dstack([xn, np.ones((H, W))])
    rlen = np.linalg.norm(ray, axis=2)
    ray /= rlen[..., None]
    R_wd = np.array(rig['view']['c2w_depth_opencv'])[:3, :3]
    n_cam = nrm @ R_wd                                     # 世界法线 → 深度相机系
    cos = np.clip(np.abs((n_cam * ray).sum(2)), 0, 1)
    rng_m = z * rlen
    sky = (z <= 0) | (z > 1e3)
    sky_geo = sky.copy()

    # Cycles 辐亮度换算到 ρ·cos/r² 单位：用纯漫反射直接光拟合比例
    lum = lambda a: lin(a) @ LUM                           # noqa: E731
    dcol, ddir = near(p['DiffCol']) @ LUM, lum(p['DiffDir'])
    analytic = cos / np.maximum(rng_m, 1e-3) ** 2
    good = mask & ~sky & (ddir > 0) & (dcol > 0.05) & (cos > 0.3)
    scale = np.median(analytic[good] / ddir[good])
    diffuse = (lin(p['DiffCol'] * (p['DiffDir'] + p['DiffInd'])) @ LUM)
    other = (lin(p['GlossCol'] * (p['GlossDir'] + p['GlossInd']) + p['TransCol'] * (p['TransDir'] + p['TransInd'])) @ LUM)

    # 850 nm 反射率修正（只作用于漫反射部分）
    table = rho_ir_table(labels, max(int(idx.max()), 1))
    rule = table[np.clip(idx, 0, len(table) - 1)]
    rho_ir = np.where(rule > 0, rule, np.where(rule < 0, np.maximum(dcol, -rule), dcol))
    ratio = np.where(dcol > 1e-3, rho_ir / np.maximum(dcol, 1e-3), 1.0)
    I = scale * (diffuse * ratio + other)

    k = cal['k']
    S = k * E * I
    s_ref = k * 0.5
    reason = np.zeros((H, W), np.uint8)

    # 玻璃：测到后方表面；信号按后方 ρ≈0.35 正对估计并乘双程透过率；近乎正对时镜面反射过曝
    glass = np.isin(idx, rig['glass_ids']) & mask
    z_meas = np.where(glass, z_through, z)
    r_bg = z_through * rlen
    S = np.where(glass, k * E * 0.35 / np.maximum(r_bg, 1e-3) ** 2 * PARAMS['glass_transmission'] ** 2, S)
    specular = glass & (cos > PARAMS['specular_cos'])

    # 镜子：测到虚像；Cycles 点光源无法经镜面照亮物体（焦散），信号按展开路程解析估计
    mirror = np.zeros((H, W), bool)
    if (view / 'mirror.npz').exists():
        mz = np.load(view / 'mirror.npz')
        nb = near(mz['bounces'].astype(np.float32))
        mirror = (nb > 0) & mask
        m_r = near(mz['range'])
        escaped = mirror & ~np.isfinite(m_r)                  # 反射后射向室外/无穷远：无回波
        m_r = np.where(np.isfinite(m_r), m_r, 0)
        z_meas = np.where(mirror, m_r / rlen, z_meas)
        S_m = k * E * PARAMS['mirror_rho'] * PARAMS['mirror_reflectance'] ** (2 * nb) * near(mz['cos_end']) / np.maximum(m_r, 1e-3) ** 2
        S = np.where(mirror, np.where(escaped, 0, S_m), S)
        specular |= mirror & (nb == 1) & (near(mz['cos_mirror']) > PARAMS['specular_cos'])
        sky = (sky & ~mirror) | escaped
    reason[specular] = 6

    # 内凹棱角偏差（镜子、玻璃除外）
    z_meas = z_meas + corner_bias(z, xn, n_cam, ray, cos, mask & ~sky & ~glass & ~mirror, K[0, 0])
    z_clean = np.where(sky, 0, z_meas).astype(np.float32)     # 加噪前，用于判断边缘

    # 随机噪声
    sigma = np.sqrt(PARAMS['sigma0_mm'] ** 2 + PARAMS['sigma1_mm'] ** 2 * s_ref / np.maximum(S, 1e-6)) / 1000
    z_meas = z_meas + rng.normal(size=z.shape) * sigma

    # 无效像素
    p_valid = 1 / (1 + (PARAMS['low_signal_ir50'] / np.maximum(S, 1e-6)) ** PARAMS['low_signal_slope'])
    reason[(rng.random(z.shape) >= p_valid) & (reason == 0)] = 2
    reason[(S > PARAMS['saturation']) & (reason == 0)] = 3
    zf = z_clean
    # 二阶差分：平面（含掠射角）≈ 0，只有深度断层才超阈值；天空按极远处理
    zs = np.pad(np.where(sky, 1e3, zf), 1, mode='edge')
    lap = np.zeros_like(zf)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        fwd = zs[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        bwd = zs[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        lap = np.maximum(lap, np.abs(2 * zs[1:-1, 1:-1] - fwd - bwd))
    edge = lap > 2 * np.maximum(PARAMS['edge_jump'][0], PARAMS['edge_jump'][1] * zf)
    dist_e = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    a, b = PARAMS['edge_halfwidth_px']
    wnoise = cv2.GaussianBlur(rng.normal(size=z.shape).astype(np.float32), (0, 0), 6)
    wnoise *= PARAMS['edge_width_noise_px'] / max(float(wnoise.std()), 1e-6)
    halfw = a + b * zf + wnoise + rng.normal(size=z.shape) * PARAMS['edge_jitter_px']
    reason[(dist_e <= halfw) & (reason == 0)] = 4
    reason[sky & (reason == 0)] = 2
    reason[~mask] = 1

    depth = np.where(reason == 0, np.round(z_meas * 1000), 0).clip(0, 65535).astype(np.uint16)
    gt = np.where(mask & ~sky_geo, np.round(z * 1000), 0).clip(0, 65535).astype(np.uint16)
    ir = S + rng.normal(size=S.shape) * np.sqrt(PARAMS['ir_gain'] * np.maximum(S, 0) + PARAMS['ir_read_noise'] ** 2)
    ir = np.where((ir > PARAMS['saturation']) | specular, 65535, ir.clip(0, PARAMS['saturation']))   # 过曝读数 65535
    ir = np.where(mask, ir, 0).astype(np.uint16)

    # 投影到彩色相机（z-buffer 取最近）
    Kc = np.array(rig['color']['K'])
    Wc, Hc = rig['color']['size']
    T = np.array(rig['T_color_from_depth'])
    v = depth > 0
    zz = depth[v] / 1000.0
    P = np.stack([xn[..., 0][v] * zz, xn[..., 1][v] * zz, zz], -1) @ T[:3, :3].T + T[:3, 3]
    u_c = np.round(Kc[0, 0] * P[:, 0] / P[:, 2] + Kc[0, 2]).astype(int)
    v_c = np.round(Kc[1, 1] * P[:, 1] / P[:, 2] + Kc[1, 2]).astype(int)
    ok = (u_c >= 0) & (u_c < Wc) & (v_c >= 0) & (v_c < Hc) & (P[:, 2] > 0)
    d2c = np.full(Hc * Wc, np.inf)
    np.minimum.at(d2c, v_c[ok] * Wc + u_c[ok], P[ok, 2] * 1000)
    d2c = np.where(np.isfinite(d2c), np.round(d2c), 0).reshape(Hc, Wc).astype(np.uint16)

    for name, img in (('depth', depth), ('ir', ir), ('depth_gt', gt), ('depth_to_color', d2c), ('invalid_reason', reason)):
        cv2.imwrite(str(view / f'{name}.png'), img)
    err = (depth.astype(np.float32) - gt) * (depth > 0)
    inm = mask & ~sky
    stats = {'valid_in_mask': float((depth > 0)[inm].mean()),
             'reasons_in_mask': {str(c): float((reason[inm] == c).mean()) for c in (2, 3, 4, 6)},
             'mirror_px': int(mirror.sum()), 'mirror_valid': float((depth > 0)[mirror].mean()) if mirror.any() else None,
             'abs_err_median_mm': float(np.median(np.abs(err[depth > 0]))),
             'sigma_median_mm': float(np.median(sigma[depth > 0]) * 1000), 'cycles_scale': float(scale)}
    return stats, dict(depth=depth, gt=gt, ir=ir, reason=reason, err=err, mask=mask)


def colour_depth(d, lo=500, hi=5000):
    t = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d == 0] = 0
    return img


def tile(img, w=400, h=360):
    s = min(w / img.shape[1], h / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    canvas[y:y + im.shape[0], x:x + im.shape[1]] = im if im.ndim == 3 else im[..., None]
    return canvas


def preview(out, results, real_root):
    reason_col = np.array([[0, 0, 0], [40, 40, 40], [200, 60, 200], [255, 255, 255], [0, 200, 255],
                           [0, 140, 255], [255, 120, 0]], np.uint8)     # BGR
    rows, titles = [], []
    for name, (stats, r) in results.items():
        colour = cv2.imread(str(out / name / 'color.png'))
        irv = cv2.cvtColor(np.clip((r['ir'] / 1500.0) ** 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        e = np.clip(r['err'] / 20.0, -1, 1)
        errv = np.dstack([np.clip(-e, 0, 1), 1 - np.abs(e), np.clip(e, 0, 1)])
        errv = (np.where((r['depth'] > 0)[..., None], 0.25 + 0.75 * errv, 0) * 255).astype(np.uint8)
        rv = reason_col[r['reason']]
        rv[(r['reason'] == 0) & r['mask']] = (90, 90, 90)
        rows.append([tile(colour), tile(irv), tile(colour_depth(r['gt'])), tile(colour_depth(r['depth'])), tile(errv), tile(rv)])
        titles.append(f"{name}: valid {stats['valid_in_mask']:.0%}, median |err| {stats['abs_err_median_mm']:.1f} mm")
    # 真机参考：test 序列（办公室，近静止）同一帧的彩色、红外、深度
    # 导出目录里没有 test 的红外，直接从录像 MKV 的 IR 轨（第 3 路流）解码同一帧；与模拟红外同一显示比例
    real = sorted((real_root / 'test_pyk4a_export' / 'depth_raw').glob('*.npy'))
    if real:
        fi = 50
        d = np.load(real[fi])
        rgb = cv2.imread(str(sorted((real_root / 'test_pyk4a_export' / 'rgb_raw').glob('*.png'))[fi]))
        mkv = real_root.parent / 'video' / 'test_pyk4a.mkv'
        irr = np.zeros_like(d, np.float32)
        if mkv.exists():
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(mkv), '-map', '0:2', '-vf', f'select=eq(n\\,{fi})',
                                  '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'],
                                 capture_output=True, check=True).stdout
            irr = np.frombuffer(raw, np.uint16).reshape(d.shape).astype(np.float32)
        irr = cv2.cvtColor(np.clip((irr / 1500.0) ** 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        blank = np.zeros((360, 400, 3), np.uint8)
        rows.append([tile(rgb), tile(irr), blank, tile(colour_depth(d)), blank, blank])
        titles.append(f'REAL Azure Kinect DK (our device), office, same frame: RGB / IR / depth, invalid {(d == 0)[np.load(HERE / "dk_params.npz")["mask"]].mean():.0%} in mask')
    head = ['DK colour (720p)', 'Simulated IR', 'GT depth', 'Simulated depth', 'Error (±20 mm)', 'Invalid reason']
    th, rh = 34, 26
    sheet = np.full((th + len(rows) * (360 + rh) + 40, 2400, 3), 24, np.uint8)
    for i, row in enumerate(rows):
        y = th + i * (360 + rh) + rh
        sheet[y:y + 360] = np.hstack(row)
    im = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    f, fs = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16)
    for j, t in enumerate(head):
        dr.text((j * 400 + 8, 6), t, font=f, fill=(240, 240, 240))
    for i, t in enumerate(titles):
        dr.text((8, th + i * (360 + rh) + 4), t, font=fs, fill=(220, 220, 220))
    dr.text((8, im.height - 30), 'Depth colormap 0.5-5 m (turbo). Error: blue = too near, red = too far. Reasons: grey valid, '
            'magenta low signal, white saturated, yellow edge, blue glass/mirror specular hotspot.', font=fs, fill=(200, 200, 200))
    im.save(out / 'dk_preview.jpg', quality=90)
    print('PREVIEW', out / 'dk_preview.jpg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--real', default='/home/tcy/Programs/AzureKinectDK_Ubuntu26/output')
    opt = ap.parse_args()
    out = Path(opt.out)
    rig = json.loads((out / 'rig.json').read_text())
    labels = json.loads((out / 'labels.json').read_text()) if (out / 'labels.json').exists() else {}
    npz = np.load(HERE / 'dk_params.npz')
    cal = {'mask': npz['mask'], 'falloff': npz['falloff'], 'k': json.loads((HERE / 'dk_params.json').read_text())['ir_scale_k']}
    rng = np.random.default_rng(opt.seed)
    results = {}
    for name, view in rig['views'].items():
        if not (out / name / 'ir_passes.npz').exists():
            continue
        stats, r = simulate(out / name, dict(rig, view=view), cal, labels, rng)
        results[name] = (stats, r)
        print('SIM', name, json.dumps(stats), flush=True)
    (out / 'sim_params.json').write_text(json.dumps({'params': PARAMS, 'seed': opt.seed, 'ir_scale_k': cal['k'],
                                                      'stats': {k: v[0] for k, v in results.items()}}, indent=1))
    if opt.preview:
        preview(out, results, Path(opt.real))


if __name__ == '__main__':
    main()
