import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, least_squares
from scipy.optimize import curve_fit

# tlinterのインポート
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# cd C:\DATA_HK\python\HODACA_calibration

#windowの作成
root=tk.Tk()
root.withdraw()

# ガウス関数
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# 複数ファイル選択ダイアログ
file_paths = filedialog.askopenfilenames(title="ファイルを選択してください")

# プロット用設定
fig, axs = plt.subplots(6, 4, figsize=(12, 8))
axs = axs.flatten()

fit_results = []  # 出力内容をためておくリスト

for i in range(len(file_paths)):
    with open(file_paths[i], "r", encoding="utf-8") as f:
        lines = f.readlines()
        head = lines[31]
        head2 = head.split()

        if "Pt." in head2:
            pass
        else:
            head = lines[32]
            head2 = head.split()

    # 数値データ読み込み
    rdb = np.loadtxt(file_paths[i], comments='#')

    # パラメータ名と対応検出器を抽出
    param = head2[2]  # 左から3番目
    detector_num = param.split('-')[1]
    detector_id = f'D{detector_num}'
    detector_index = head2.index(detector_id)

    # x, y データ
    x = rdb[:, 1]  # 3列目（インデックス2） → c3
    y = rdb[:, detector_index-1]
    yerr = np.sqrt(y)

    # ガウスフィッティング
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x[np.argmax(y)], 1.0])
        mu = popt[1]
        mu_str = f"{mu:.4f}"
    except RuntimeError:
        popt = [0, 0, 0]
        mu = np.nan
        mu_str = f"{mu:.4f}"

    # プロット
    ax = axs[i]
    ax.errorbar(x, y, yerr=yerr, fmt='o', label='Data')
    if not np.isnan(mu):
        xfit = np.linspace(min(x), max(x), 500)
        yfit = gaussian(xfit, *popt)
        ax.plot(xfit, yfit, 'r--', label='Fit')
        title = f"No.{i+1} μ={mu:.4f}"
        # コンソール出力
        #print(f"drive c3-{detector_num} {mu_str}")
    else:
        title = f"No.{i+1} (Fit failed)"
    ax.set_title(title)
    ax.set_xlabel(param)
    ax.set_ylabel(detector_id)
    ax.legend()
    
    # 出力用に記録
    result_line = f"drive c3-{detector_num} {mu_str}"
    fit_results.append(result_line)

# --- 最後に保存 ---
with open("fit_results.txt", "w", encoding="utf-8") as f:
    for line in fit_results:
        f.write(line + "\n")

plt.tight_layout()
plt.show()
        
