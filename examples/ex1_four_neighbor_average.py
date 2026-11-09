"""例 1：四邻平均 ↔ 拉普拉斯方程

对一个函数 V(x, y)，在点 (x0, y0) 处比较
    四邻平均 = [V(x0+h,y0) + V(x0-h,y0) + V(x0,y0+h) + V(x0,y0-h)] / 4
和中心值 V(x0, y0)。

泰勒展开给出：四邻平均 - 中心值 = (h^2 / 4) * ∇²V + O(h^4)
所以 "中心值 = 四邻平均" 恰好就是离散版的 ∇²V = 0。
"""
import numpy as np


def avg_minus_center(V, x0, y0, h):
    avg = (V(x0 + h, y0) + V(x0 - h, y0) + V(x0, y0 + h) + V(x0, y0 - h)) / 4
    return avg - V(x0, y0)


tests = {
    "x^2 - y^2   (∇²V = 0)": (lambda x, y: x**2 - y**2, 0.0),
    "x^2 + y^2   (∇²V = 4)": (lambda x, y: x**2 + y**2, 4.0),
    "e^x sin y   (∇²V = 0)": (lambda x, y: np.exp(x) * np.sin(y), 0.0),
    "x^4         (∇²V = 12x^2)": (lambda x, y: x**4, None),
}

x0, y0 = 0.7, 0.3
for name, (V, lap) in tests.items():
    print(name)
    for h in [0.1, 0.05, 0.025]:
        d = avg_minus_center(V, x0, y0, h)
        lap_true = 12 * x0**2 if lap is None else lap
        print(f"   h={h:<6} 平均-中心={d: .3e}   h²/4·∇²V={h*h/4*lap_true: .3e}")
