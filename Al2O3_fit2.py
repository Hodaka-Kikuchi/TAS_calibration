import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, basinhopping

# 六方晶 Al₂O₃ の格子定数
a_Al2O3 = 4.758  # [Å]
c_Al2O3 = 12.991  # [Å]
lambda_ideal = 2.372  # [Å] 理想的な波長
d_mono = 3.355  # モノクロメータの d 値 [Å]

# CSV データ (h, k, l, 2θ) を NumPy 配列として読み込む
"""
# 理想値
data = np.array([
    [0.00, 1.00, 2.00, 39.8498],
    [1.00, 0.00, 4.00, 55.4031],
    [0.00, 0.00, 6.00, 66.4082],
    [1.00, 1.00, 3.00, 69.3191],
    [0.00, 2.00, 4.00, 85.9339],
    [1.00, 1.00, 6.00, 95.5477]
])
""" 
# 12.74 meVすなわち2.5341A、差は-0.1621A
data = np.array([
    [0.00, 1.00, 2.00, 42.7069],
    [1.00, 0.00, 4.00, 59.5647],
    [1.00, 1.00, 3.00, 74.8370],
    [0.00, 2.00, 4.00, 93.4782],
    [1.00, 1.00, 6.00, 104.5982]
])

h, k, l, theta_obs = data.T
theta_obs = np.radians(theta_obs / 2)  # 2θ → θ に変換

# 六方晶の d_hkl の計算
def calc_d_hkl(h, k, l, a, c):
    term1 = (4/3) * (h**2 + h*k + k**2) / a**2
    term2 = (l**2) / c**2
    return 1 / np.sqrt(term1 + term2)

d_hkl = calc_d_hkl(h, k, l, a_Al2O3, c_Al2O3)

# フィッティング関数 (2θ のオフセット考慮)
def bragg_fit(d, offset_2theta):
    return np.arcsin(lambda_ideal / (2 * d)) * 2 + offset_2theta

# 初期値の設定
offset_2theta_init = 0.0  # まずは 0 に設定

# ローカルフィット (初期値を得る)
params, covariance = curve_fit(bragg_fit, d_hkl, theta_obs * 2, p0=[offset_2theta_init], method='trf')
offset_2theta_local = params[0]  # ローカルフィットの結果

# グローバル最適化用の関数
def cost_function(offset_2theta):
    theta_fit = bragg_fit(d_hkl, offset_2theta)
    return np.sum((theta_fit - theta_obs * 2) ** 2)  # 残差の二乗和を最小化

# グローバル最適化 (basinhopping)
minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(-5, 5)]}  # 探索範囲を ±5° に制限
opt_result = basinhopping(cost_function, offset_2theta_local, minimizer_kwargs=minimizer_kwargs, niter=100)

# 最終的なオフセット値
offset_2theta_fit = opt_result.x[0]

# フィッティング後の 2θ と λ
theta_fit = np.arcsin(lambda_ideal / (2 * d_hkl)) * 2 + offset_2theta_fit
lambda_fit = 2 * d_hkl * np.sin(theta_fit / 2)
delta_lambda = lambda_ideal - np.mean(lambda_fit)

# モノクロメータの回転角
theta_mono_ideal = np.arcsin(lambda_ideal / (2 * d_mono))
theta_mono_fit = np.arcsin(np.mean(lambda_fit) / (2 * d_mono))
delta_theta_mono = np.degrees(theta_mono_ideal - theta_mono_fit)

# 結果の表示
print(f"フィッティング結果:")
print(f"  ローカルフィットの 2θ オフセット = {np.degrees(offset_2theta_local):.4f}°")
print(f"  グローバルフィットの 2θ オフセット = {np.degrees(offset_2theta_fit):.4f}°")
print(f"  波長のずれ Δλ = {delta_lambda:.6f} Å")
print(f"  モノクロメータの回転角のずれ Δθ_mono = {delta_theta_mono:.4f}°")

# プロット
plt.scatter(np.degrees(theta_obs * 2), np.degrees(theta_fit), label="Fitted", color="red")
plt.plot([30, 110], [30, 110], linestyle="--", color="black", label="y=x (ideal)")
plt.xlabel("Observed 2θ [deg]")
plt.ylabel("Fitted 2θ [deg]")
plt.legend()
plt.grid()
plt.show()
