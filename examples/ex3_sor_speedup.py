"""例 3：Jacobi / Gauss-Seidel / SOR 收敛速度对比（一维玩具版）

Gauss-Seidel：算出一个新值后立刻用它（原地更新）。
SOR：在 Gauss-Seidel 的更新量上乘以 ω (1<ω<2)，"多走一点"。
一维 / 方形区域的最优 ω ≈ 2 / (1 + sin(π/N))。
"""
import numpy as np


def solve(N, method, omega=1.0, tol=1e-8, max_steps=200000):
    x = np.linspace(0, 1, N + 1)
    V = np.zeros(N + 1)
    V[-1] = 1.0
    for step in range(1, max_steps + 1):
        if method == "jacobi":
            V[1:-1] = 0.5 * (V[:-2] + V[2:])
        else:  # gauss-seidel / sor：逐点原地更新
            for i in range(1, N):
                gs = 0.5 * (V[i - 1] + V[i + 1])
                V[i] += omega * (gs - V[i])
        if np.max(np.abs(V - x)) < tol:
            return step
    return None


for N in [20, 40, 80]:
    w_opt = 2 / (1 + np.sin(np.pi / N))
    print(f"N={N:3d}: Jacobi {solve(N,'jacobi'):6d} 步 | "
          f"Gauss-Seidel {solve(N,'gs'):6d} 步 | "
          f"SOR(ω={w_opt:.3f}) {solve(N,'sor',w_opt):4d} 步")
