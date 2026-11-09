"""例 4：验证接地导体球在匀强电场中的解析解

    V(r, θ) = -E0 (r - R^3 / r^2) cos θ        (r ≥ R)

要检查三件事：
  (1) 球外满足 ∇²V = 0
  (2) 球面 r = R 上 V = 0（接地）
  (3) r → ∞ 时 V → -E0 z（回到原来的匀强场）
这里用有限差分在直角坐标下数值检查 (1)，不依赖任何符号推导。
"""
import numpy as np

E0, R = 1.0, 1.0


def V(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    return -E0 * z * (1 - R**3 / r**3)  # cosθ = z/r，所以 r cosθ = z


def laplacian(f, x, y, z, h=1e-3):
    c = f(x, y, z)
    return (f(x + h, y, z) + f(x - h, y, z) + f(x, y + h, z) + f(x, y - h, z)
            + f(x, y, z + h) + f(x, y, z - h) - 6 * c) / h**2


rng = np.random.default_rng(0)
print("(1) 球外随机点的 ∇²V（应 ≈ 0，残差来自差分误差）：")
for _ in range(4):
    p = rng.uniform(-3, 3, size=3)
    if np.linalg.norm(p) > 1.2:
        print(f"   r={np.linalg.norm(p):.2f}   ∇²V = {laplacian(V, *p): .2e}")

print("(2) 球面上 V：", [f"{V(*(R*np.array([np.sin(t), 0, np.cos(t)]))):.1e}"
                         for t in np.linspace(0, np.pi, 5)])

print("(3) 远处 V / (-E0 z)：", [f"{V(0, 0, z)/(-E0*z):.6f}" for z in [5, 20, 100]])

# 附：面电荷 σ = -ε0 ∂V/∂r |_{r=R} = 3 ε0 E0 cosθ
h = 1e-6
for t in [0, np.pi / 3, np.pi / 2]:
    n = np.array([np.sin(t), 0, np.cos(t)])
    dVdr = (V(*((R + h) * n)) - V(*(R * n))) / h
    print(f"   θ={np.degrees(t):5.1f}°  -∂V/∂r = {-dVdr:.4f}   3E0cosθ = {3*E0*np.cos(t):.4f}")
