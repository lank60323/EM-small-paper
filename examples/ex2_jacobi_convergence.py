"""例 2：为什么迭代会收敛、收敛有多快（一维玩具版）

一维拉普拉斯方程 V'' = 0，两端 V(0)=0、V(1)=1，精确解是直线 V = x。
Jacobi 迭代：每个内点替换为左右邻居的平均。

误差 e = V_迭代 - V_精确 满足同样的平均规则，但两端为 0。
每一步误差大约乘以谱半径 ρ = cos(π/N)，N 为格子数。
ρ 越接近 1 收敛越慢：ρ ≈ 1 - π²/(2N²)，所以需要的步数 ∝ N²。
"""
import numpy as np


def jacobi_1d(N, steps):
    x = np.linspace(0, 1, N + 1)
    V = np.zeros(N + 1)
    V[-1] = 1.0
    errs = []
    for _ in range(steps):
        V[1:-1] = 0.5 * (V[:-2] + V[2:])  # 右边先整体算好再赋值 → 这就是 Jacobi
        errs.append(np.max(np.abs(V - x)))
    return np.array(errs)


for N in [10, 20, 40]:
    errs = jacobi_1d(N, 20000)
    # 取误差在 1e-3 ~ 1e-9 之间的一段（避开开头的过渡和末尾的机器精度），求平均每步缩小比
    k1, k2 = np.argmax(errs < 1e-3), np.argmax(errs < 1e-9)
    ratio = (errs[k2] / errs[k1]) ** (1 / (k2 - k1))
    steps_to_1e6 = np.argmax(errs < 1e-6) + 1
    print(f"N={N:3d}  实测每步缩小比 {ratio:.6f}   理论 cos(π/N) = {np.cos(np.pi/N):.6f}"
          f"   误差降到 1e-6 需 {steps_to_1e6} 步")
