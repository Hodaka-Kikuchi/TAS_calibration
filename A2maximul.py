import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, least_squares
from scipy.optimize import curve_fit

# tlinterのインポート
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# cd C:\DATA_HK\python\HODACA_calibration

# データ
hw = np.array([7.365,7,6.865,6.5,6.365,6,5.865,5.5,5.365,5,4.865,4.365,3.865,3.365,2.865,2.365,1.865,1.365,0.865,0.365,0,-0.435])
maxA2 = np.array([34,35,36,40,42,45,46,50,54,60,64,68,72,76,82,86,90,94,100,101,105,107])
"""
# 直線fitと２次関数fitして図示
# 線形フィット（1次関数）
coeffs_linear = np.polyfit(hw, maxA2, 1)
linear_fit = np.poly1d(coeffs_linear)

# 2次関数フィット（2次多項式）
coeffs_quad = np.polyfit(hw, maxA2, 2)
quad_fit = np.poly1d(coeffs_quad)

# プロット用x値生成（滑らかな曲線）
x_fit = np.linspace(min(hw), max(hw), 300)

# プロット
plt.figure(figsize=(8, 6))
plt.scatter(hw, maxA2, color='black', label='Data')
plt.plot(x_fit, linear_fit(x_fit), 'r--', label=f'Linear Fit: y = {coeffs_linear[0]:.4f}x + {coeffs_linear[1]:.4f}')
plt.plot(x_fit, quad_fit(x_fit), 'b-', label=f'Quadratic Fit: y = {coeffs_quad[0]:.4f}x² + {coeffs_quad[1]:.4f}x + {coeffs_quad[2]:.4f}')

plt.xlabel('hw')
plt.ylabel('maxA2')
plt.title('Linear and Quadratic Fit of Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# フィルタ分け（x=5 を境に2分割）
hw_high = hw[hw >= 5]
maxA2_high = maxA2[hw >= 5]

hw_low = hw[hw < 5]
maxA2_low = maxA2[hw < 5]

# 線形フィット（それぞれ別に）
coeff_high = np.polyfit(hw_high, maxA2_high, 1)
fit_high = np.poly1d(coeff_high)

coeff_low = np.polyfit(hw_low, maxA2_low, 1)
fit_low = np.poly1d(coeff_low)

# 描画用x範囲
x_high = np.linspace(min(hw_high), max(hw_high), 200)
x_low = np.linspace(min(hw_low), max(hw_low), 200)

# プロット
plt.figure(figsize=(8, 6))
plt.scatter(hw, maxA2, color='black', label='Data')

# フィット曲線
plt.plot(x_high, fit_high(x_high), 'r--', label=f'Fit (hw ≥ 5): y={coeff_high[0]:.3f}x+{coeff_high[1]:.3f}')
plt.plot(x_low, fit_low(x_low), 'b--', label=f'Fit (hw < 5): y={coeff_low[0]:.3f}x+{coeff_low[1]:.3f}')

# 軸ラベルなど
plt.xlabel('hw')
plt.ylabel('maxA2')
plt.title('Piecewise Linear Fit (Split at hw = 5)')
plt.axvline(5, color='gray', linestyle=':', label='hw = 5 split')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()