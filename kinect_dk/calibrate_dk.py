"""从真机 Azure Kinect DK 录制数据标定模拟参数，输出 dk_params.npz / dk_params.json。
用 anaconda Python（numpy + cv2）运行：python3 calibrate_dk.py [导出目录根]
- 有效区域 mask：所有帧中曾出现有效深度的像素（NFOV 的六边形照明区）
- 光源照度分布 E(u,v)：dry 序列近似平整的墙面，IR × r² / cos 后归一化并平滑
- 亮度比例 k：假设墙面红外反射率 WALL_RHO，使 IR = k · ρ · cos / r² · E（r 单位米）
- 时间噪声：test 序列（近静止）按距离分桶的逐像素深度标准差，供核对噪声模型
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/tcy/Programs/AzureKinectDK_Ubuntu26/output')
HERE = Path(__file__).resolve().parent
WALL_RHO = 0.6   # 米色抹灰墙在 850 nm 的反射率假设值


def load_seq(name, kind, step=1):
    return [np.load(f) for f in sorted(glob.glob(str(ROOT / name / kind / '*.npy')))[::step]]


def main():
    K = np.loadtxt(ROOT / 'dry_pyk4a_export' / 'intrinsic_depth.txt')
    dist = np.loadtxt(ROOT / 'dry_pyk4a_export' / 'distortion_depth.txt')
    depths = load_seq('dry_pyk4a_export', 'depth_raw', 5) + load_seq('wet_pyk4a_export', 'depth_raw', 5)
    mask = np.any([d > 0 for d in depths], axis=0)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)

    # 墙面：逐帧拟合平面，求 r 与入射角余弦，E ∝ IR·r²/cos
    acc = []
    irs = load_seq('dry_pyk4a_export', 'ir_raw', 10)
    pcs = load_seq('dry_pyk4a_export', 'point_cloud', 10)
    for ir, pc in zip(irs, pcs):
        P = pc.reshape(-1, 3).astype(np.float64) / 1000
        ok = (P[:, 2] > 0) & (ir.reshape(-1) > 0)
        A = np.c_[P[ok, 0], P[ok, 1], np.ones(ok.sum())]
        for _ in range(3):   # 迭代剔除孔洞等离群点
            coef, *_ = np.linalg.lstsq(A, P[ok, 2], rcond=None)
            res = np.abs(A @ coef - P[ok, 2])
            keep = res < max(0.004, 3 * np.median(res))
            A, idx = A[keep], np.flatnonzero(ok)[keep]
            ok = np.zeros_like(ok)
            ok[idx] = True
        n = np.array([coef[0], coef[1], -1.0])
        n /= np.linalg.norm(n)
        r = np.linalg.norm(P, axis=1)
        cos = np.abs(P @ n) / np.maximum(r, 1e-6)
        e = np.full(P.shape[0], np.nan)
        e[ok] = ir.reshape(-1)[ok] * r[ok] ** 2 / cos[ok]
        acc.append(e.reshape(ir.shape))
    E = np.nanmedian(np.stack(acc), axis=0)
    h, w = E.shape
    cy, cx = int(K[1, 2]), int(K[0, 2])
    centre = np.nanmedian(E[cy - 20:cy + 20, cx - 20:cx + 20])
    k = centre / WALL_RHO
    # 平滑：在 mask 内用归一化卷积补洞后高斯平滑
    valid = np.isfinite(E) & mask
    En = np.where(valid, E / centre, 0).astype(np.float32)
    wgt = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 15)
    En = cv2.GaussianBlur(En, (0, 0), 15) / np.maximum(wgt, 1e-3)
    # 再用 4 阶二维多项式拟合，去掉墙面纹理残留，只保留低频照度分布
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - cx) / w, (yy - cy) / h
    terms = [u ** i * v ** j for i in range(5) for j in range(5 - i)]
    fit = valid & (wgt > 0.5)
    coef, *_ = np.linalg.lstsq(np.stack([t[fit] for t in terms], 1), En[fit], rcond=None)
    En = sum(c * t for c, t in zip(coef, terms))
    En = np.where(mask, np.clip(En / En[cy, cx], 0.05, None), 0).astype(np.float32)

    # 时间噪声（test 近静止序列）
    D = np.stack(load_seq('test_pyk4a_export', 'depth_raw')).astype(np.float32)
    allv = (D > 0).all(0)
    m, s = D.mean(0), D.std(0)
    noise = []
    for lo, hi in [(400, 700), (700, 1000), (1000, 1500), (1500, 2500), (2500, 4000), (4000, 6000)]:
        sel = allv & (m >= lo) & (m < hi)
        if sel.sum() > 200:
            noise.append({'range_mm': [lo, hi], 'n': int(sel.sum()), 'std_median_mm': float(np.median(s[sel]))})
    flicker = float(((D > 0).any(0) & ~allv)[mask].mean())

    np.savez_compressed(HERE / 'dk_params.npz', mask=mask, falloff=En, K=K, dist=dist)
    info = {'K_depth': K.tolist(), 'dist_depth_k1k2p1p2k3k4k5k6': dist.tolist(), 'size': [w, h],
            'ir_scale_k': float(k), 'wall_rho_assumed': WALL_RHO, 'mask_fraction': float(mask.mean()),
            'falloff_edge_p10': float(np.percentile(En[mask], 10)), 'temporal_noise_test': noise,
            'flicker_fraction_in_mask_test': flicker}
    (HERE / 'dk_params.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    vis = cv2.applyColorMap((np.clip(En, 0, 1.2) / 1.2 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis[~mask] = 0
    cv2.imwrite(str(HERE / 'dk_falloff.png'), vis)


if __name__ == '__main__':
    main()
