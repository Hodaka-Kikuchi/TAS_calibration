#cd C:\DATA_HK\python\HODACA_calibration
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, least_squares

# これを使用すること

# モノクロメータの d 値 [Å]
d_mono = 3.355/2  
#ei = 5*4
#lambda_ideal=9.045/(ei**(1/2))
#print(lambda_ideal)
lambda_ideal = 2.372  # [Å] 理想波長(HODACA:2.372,HER:2.023)

# 六方晶 Al₂O₃ の格子定数
a_Al2O3 = 4.758  # [Å]
c_Al2O3 = 12.991  # [Å]

# CSV データ (h, k, l, 2θ) を NumPy 配列として読み込む
"""# 理想値(HODACA, 3.635meV,2.372Å)
data = np.array([
    [0.00, 1.00, 2.00, 39.8498],
    [1.00, 0.00, 4.00, 55.4031],
    [0.00, 0.00, 6.00, 66.4082],
    [1.00, 1.00, 3.00, 69.3191],
    [0.00, 2.00, 4.00, 85.9339],
    [1.00, 1.00, 6.00, 95.5477]
])
"""

"""# 理想値(HER, 5.000meV,2.023Å)
data = np.array([
    [1.00,1.00,3.00,58.010],
    [0.00,2.00,4.00,71.060],
    [1.00,1.00,6.00,78.302],
    [3.00,0.00,0.00,94.800]
])
"""

"""# A2のみずらす。
data = np.array([
    [0.00, 1.00, 2.00, 38.8498],
    [1.00, 0.00, 4.00, 54.4031],
    [0.00, 0.00, 6.00, 65.4082],
    [1.00, 1.00, 3.00, 68.3191],
    [0.00, 2.00, 4.00, 84.9339],
    [1.00, 1.00, 6.00, 94.5477]
])
"""

"""# lamdaのみずらす。12.74 meVすなわち2.5341A、差は-0.1621A
data = np.array([
    [0.00, 1.00, 2.00, 42.7069],
    [1.00, 0.00, 4.00, 59.5647],
    [1.00, 1.00, 3.00, 74.8370],
    [0.00, 2.00, 4.00, 93.4782],
    [1.00, 1.00, 6.00, 104.5982]
])
"""

"""
data = np.array([
    [1.00,1.00,3.00,58.264],
    [0.00,2.00,4.00,71.326],
    [1.00,1.00,6.00,78.561],
    [3.00,0.00,0.00,95.077]
])
"""

data = np.array([
    [0.00, 1.00, 2.00, 39.8404],
    [1.00, 0.00, 4.00, 55.4065],
    #[0.00, 0.00, 6.00, 66.4082],
    [1.00, 1.00, 3.00, 69.3545],
    [0.00, 2.00, 4.00, 85.9428],
    [1.00, 1.00, 6.00, 95.5771]
])

h, k, l, theta_obs = data.T
theta_obs = np.radians(theta_obs / 2)  # 2θ → θ に変換

# 六方晶の d_hkl の計算
def calc_d_hkl(h, k, l, a, c):
    term1 = (4/3) * (h**2 + h*k + k**2) / a**2
    term2 = (l**2) / c**2
    return 1 / np.sqrt(term1 + term2)

d_hkl = calc_d_hkl(h, k, l, a_Al2O3, c_Al2O3)

# 残差関数 (最小化する目的関数)
def residuals(params, d_hkl, theta_obs):
    delta_A1, delta_A2 = params

    # モノクロメータの回転角 (A1)
    A1 = np.arcsin(lambda_ideal / (2 * d_mono))  # 理想値
    A1_new = A1 + np.radians(delta_A1)  # ΔA1 を加えたもの

    # ΔA1 による波長の変化
    lambda_new = 2 * d_mono * np.sin(A1_new)

    # 新しい 2θ (A2) の計算
    theta_calc = np.arcsin(lambda_new / (2 * d_hkl))

    # ΔA2 (2θオフセット) を考慮
    theta_calc = theta_calc + np.radians(delta_A2 / 2)

    return np.sum((theta_calc - theta_obs) ** 2)  # 残差の 2 乗和を返す

# 初期値 (ΔA1 = 0°, ΔA2 = 0°)
initial_guess = [0.0, 0.0]

# グローバル最適化 (basinhopping)
minimizer_kwargs = {"method": "L-BFGS-B", "args": (d_hkl, theta_obs)}
result = basinhopping(residuals, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=200, T=1.0, stepsize=0.5)

# 最適化されたパラメータ
delta_A1_fit, delta_A2_fit = result.x

# フィッティング後の波長とモノクロメータ回転角
A1_ideal = np.arcsin(lambda_ideal / (2 * d_mono))
A1_fit = A1_ideal + np.radians(delta_A1_fit)
lambda_fit = 2 * d_mono * np.sin(A1_fit)

# 波長のずれとモノクロメータの回転角
delta_lambda = lambda_ideal - lambda_fit
delta_theta_mono = np.degrees(A1_ideal - A1_fit)

# 結果の表示
print(f"フィッティング結果:")
#print(f"  モノクロメータの回転角のずれ ΔA1 = {delta_A1_fit:.4f}°")
#print(f"  2θ のオフセット ΔA2 = {delta_A2_fit:.4f}°")
print(f"  波長のずれ Δλ = {delta_lambda:.6f} Å")
#print(f"  モノクロメータの回転角のずれ Δθ_mono = {delta_theta_mono:.4f}°")

print(f"  se c1 @(c1)-({delta_theta_mono:.4f})")
print(f"  se a1 @(a1)-({2*delta_theta_mono:.4f})")
print(f"  se a2 @(a2)-({(delta_A2_fit):.4f})")

# プロット
theta_fit = np.arcsin(lambda_fit / (2 * d_hkl)) + np.radians(delta_A2_fit / 2)
plt.scatter(np.degrees(theta_obs * 2), np.degrees(theta_fit * 2), label="Fitted", color="red")
plt.plot([30, 110], [30, 110], linestyle="--", color="black", label="y=x (ideal)")
plt.xlabel("Observed 2θ [deg]")
plt.ylabel("Fitted 2θ [deg]")
plt.legend()
plt.grid()
plt.show()
