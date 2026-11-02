"""Azure Kinect DK 模拟共用函数（Blender 与系统 Python 都可导入，只依赖 numpy）。"""
import numpy as np


def undistort(pts, K, dist, iters=20):
    """Brown-Conrady（OpenCV rational：k1 k2 p1 p2 k3 k4 k5 k6）迭代去畸变，返回归一化坐标。
    比 cv2.undistortPoints（默认 5 次迭代）更准：有效区域内重投影误差 < 1e-4 像素。"""
    k1, k2, p1, p2, k3, k4, k5, k6 = dist
    xd = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yd = (pts[:, 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / radial, (yd - dy) / radial
    return np.stack([x, y], -1)
