"""例 5：轴对称 (s, z) 坐标下的差分格式

二维直角网格 (x, y) 上的四邻平均，描述的是"沿 z 方向无限长"的问题，
即导体【圆柱】，不是导体球！要算球，得用柱坐标 (s, z)，利用绕 z 轴的对称性：

    ∇²V = ∂²V/∂s² + (1/s) ∂V/∂s + ∂²V/∂z² = 0

离散后（s = i·h，i ≥ 1）：
    V[i,j] = [ (1 + 1/(2i)) V[i+1,j] + (1 - 1/(2i)) V[i-1,j] + V[i,j+1] + V[i,j-1] ] / 4
在轴上 s = 0（i = 0），用洛必达 (1/s)∂V/∂s → ∂²V/∂s²：
    V[0,j] = [ 4 V[1,j] + V[0,j+1] + V[0,j-1] ] / 6

检验方法：拿一个已知满足轴对称拉普拉斯方程的函数，看格式是否"自洽"。
V = z² - s²/2 ：∂²/∂s² = -1，(1/s)∂/∂s = -1，∂²/∂z² = 2，和为 0 ✓
"""
import numpy as np

h = 0.1


def V(s, z):
    return z**2 - s**2 / 2


def V_dipole(s, z):  # z / r^3：球外的偶极子项，也满足拉普拉斯方程
    return z / (s * s + z * z) ** 1.5


for name, f in [("z² - s²/2", V), ("z/r³", V_dipole)]:
    print(name)
    z = 1.3
    for i in [0, 1, 3, 10]:
        s = i * h
        if i == 0:
            rhs = (4 * f(h, z) + f(0, z + h) + f(0, z - h)) / 6
        else:
            rhs = ((1 + 1 / (2 * i)) * f(s + h, z) + (1 - 1 / (2 * i)) * f(s - h, z)
                   + f(s, z + h) + f(s, z - h)) / 4
        naive = (f(s + h, z) + f(abs(s - h), z) + f(s, z + h) + f(s, z - h)) / 4
        print(f"   i={i:2d}  轴对称格式误差 {rhs - f(s, z): .2e}   "
              f"若误用平面四邻平均 {naive - f(s, z): .2e}")
