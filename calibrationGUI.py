#cd C:\DATA_HK\python\HODACA_calibration
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')#よくわかんないけどこれないとexe化したときにグラフが表示されない。超重要
from scipy.optimize import basinhopping, least_squares
import math
import configparser
# osのインポート
import os
import sys

# 右上にバージョン情報を表示
__version__ = '1.0.0'

# tlinterのインポート
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

#windowの作成
root=tk.Tk()
#windowのタイトル変更
root.title(f"Calibration ver: {__version__}")
# TriAxionSim: 三軸 (triple-axis) と「軌跡」や「軸」 (axion) 、Simulationを意識
#windowのサイズ指定
root.geometry("800x850")#550*840

# フォント設定
default_font_size = 10
entry_font = ('Helvetica', default_font_size)

# ×ボタンを押すと作成したグラフ全てがクリアされる。
# ウィンドウが閉じられたときの処理
def on_closing():
    # すべてのグラフウィンドウを閉じる
    plt.close('all')
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)  # ウィンドウが閉じられるときの振る舞いを指定

# rootのグリッド設定
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=2)
root.rowconfigure(1, weight=2)
root.rowconfigure(2, weight=5)
root.rowconfigure(3, weight=4)

# 格子定数入力フレーム
LC = ttk.Labelframe(root,text= "lattice constant")
LC.grid(row=0, column=0,sticky="NSEW")
#frame2cb.grid_propagate(True)

# グリッドの設定（列と行の重みを均等にする）
for i in range(6):  # 0-6列までの設定
    LC.columnconfigure(i, weight=1)
for i in range(2):  # 0-2行までの設定
    LC.rowconfigure(i, weight=1)

# 格子定数を入力する欄
lc_lbl1 = tk.Label(LC,text='a')
lc_lbl1.grid(row=0, column=0,sticky="NSEW")
lc_lbl2 = tk.Label(LC,text='b')
lc_lbl2.grid(row=0, column=1,sticky="NSEW")
lc_lbl3 = tk.Label(LC,text='c')
lc_lbl3.grid(row=0, column=2,sticky="NSEW")
lc_lbl4 = tk.Label(LC,text='alpha')
lc_lbl4.grid(row=0, column=3,sticky="NSEW")
lc_lbl5 = tk.Label(LC,text='beta')
lc_lbl5.grid(row=0, column=4,sticky="NSEW")
lc_lbl6 = tk.Label(LC,text='ganma')
lc_lbl6.grid(row=0, column=5,sticky="NSEW")

lc_txt1 = ttk.Entry(LC)
lc_txt1.grid(row=1, column=0,sticky="NSEW")
#lc_txt1.insert(0,'4.758')
lc_txt2 = ttk.Entry(LC)
lc_txt2.grid(row=1, column=1,sticky="NSEW")
#lc_txt2.insert(0,'4.758')
lc_txt3 = ttk.Entry(LC)
lc_txt3.grid(row=1, column=2,sticky="NSEW")
#lc_txt3.insert(0,'12.991')
lc_txt4 = ttk.Entry(LC)
lc_txt4.grid(row=1, column=3,sticky="NSEW")
#lc_txt4.insert(0,'90')
lc_txt5 = ttk.Entry(LC)
lc_txt5.grid(row=1, column=4,sticky="NSEW")
#lc_txt5.insert(0,'90')
lc_txt6 = ttk.Entry(LC)
lc_txt6.grid(row=1, column=5,sticky="NSEW")
#lc_txt6.insert(0,'120')

# 測定条件入力フレーム
MC = ttk.Labelframe(root,text= "measurement condition")
MC.grid(row=1, column=0,sticky="NSEW")

# グリッドの設定（列と行の重みを均等にする）
for i in range(3):  # 0-6列までの設定
    MC.columnconfigure(i, weight=1)
for i in range(2):  # 0-2行までの設定
    MC.rowconfigure(i, weight=1)

# 測定条件を入力する欄
mc_lbl1 = tk.Label(MC,text='E (meV)')
mc_lbl1.grid(row=0, column=0,sticky="NSEW")
mc_lbl2 = tk.Label(MC,text='λ (Å)')
mc_lbl2.grid(row=0, column=1,sticky="NSEW")
mc_lbl3 = tk.Label(MC,text='k (Å⁻¹)')
mc_lbl3.grid(row=0, column=2,sticky="NSEW")

mc_txt1 = ttk.Entry(MC)
mc_txt1.grid(row=1, column=0,sticky="NSEW")
#mc_txt1.insert(0,'14.5404')
mc_txt2 = ttk.Entry(MC)
mc_txt2.grid(row=1, column=1,sticky="NSEW")
#mc_txt2.insert(0,'2.3720')
mc_txt3 = ttk.Entry(MC)
mc_txt3.grid(row=1, column=2,sticky="NSEW")
#mc_txt3.insert(0,'2.6489')

def trans_ELK(event=None):
    """
    入力された値に基づいてエネルギー、波長、波数を計算し、エントリーボックスを更新します。
    """
    try:
        # フォーカスされているウィジェットを特定
        focused_widget = root.focus_get()

        # 各エントリーボックスの値を取得
        energy = mc_txt1.get().strip()
        wavelength = mc_txt2.get().strip()
        wavenumber = mc_txt3.get().strip()

        # フォーカスされているエントリーボックスに応じて処理
        if focused_widget == mc_txt1 and energy:
            energy = float(energy)
            wavelength = math.sqrt(81.81 / energy)  # Å
            wavenumber = 2 * math.pi / wavelength  # Å⁻¹
            # 他のボックスをクリアして計算結果を出力
            mc_txt2.delete(0, tk.END)
            mc_txt3.delete(0, tk.END)
            mc_txt2.insert(0, f"{wavelength:.4f}")
            mc_txt3.insert(0, f"{wavenumber:.4f}")

        elif focused_widget == mc_txt2 and wavelength:
            wavelength = float(wavelength)
            energy = 81.81 / (wavelength ** 2)  # meV
            wavenumber = 2 * math.pi / wavelength  # Å⁻¹
            # 他のボックスをクリアして計算結果を出力
            mc_txt1.delete(0, tk.END)
            mc_txt3.delete(0, tk.END)
            mc_txt1.insert(0, f"{energy:.4f}")
            mc_txt3.insert(0, f"{wavenumber:.4f}")

        elif focused_widget == mc_txt3 and wavenumber:
            wavenumber = float(wavenumber)
            wavelength = 2 * math.pi / wavenumber  # Å
            energy = 81.81 / (wavelength ** 2)  # meV
            # 他のボックスをクリアして計算結果を出力
            mc_txt1.delete(0, tk.END)
            mc_txt2.delete(0, tk.END)
            mc_txt2.insert(0, f"{wavelength:.4f}")
            mc_txt1.insert(0, f"{energy:.4f}")

        else:
            pass
            return

    except ValueError:
        pass

# エンターキーで計算を実行
mc_txt1.bind("<Return>", trans_ELK)
mc_txt2.bind("<Return>", trans_ELK)
mc_txt3.bind("<Return>", trans_ELK)

# hkl入力フレーム
HKL = ttk.Labelframe(root,text= "measurement results")
HKL.grid(row=2, column=0,sticky="NSEW")
#frame2cb.grid_propagate(True)

# グリッドの設定（列と行の重みを均等にする）
for i in range(6):  # 0-5列までの設定
    HKL.columnconfigure(i, weight=1)
for i in range(7):  # 0-7行までの設定
    HKL.rowconfigure(i, weight=1)
    
# 格子定数入力フレーム
RD = ttk.Labelframe(root,text= "fitting results")
RD.grid(row=3, column=0,sticky="NSEW")

# グリッドの設定（列と行の重みを均等にする）
for i in range(3):  # 0-5列までの設定
    RD.columnconfigure(i, weight=1)
for i in range(4):  # 0-7行までの設定
    RD.rowconfigure(i, weight=1)
    
# エントリボックスをリストに追加
hkl_labels = ["h", "k", "l", "A2obs", "A2calc"]
for i, label in enumerate(hkl_labels):
    ttk.Label(HKL, text=label).grid(row=0, column=1+i, sticky="NSEW")
"""
default_hkl_values =  np.array([
    [0, 1, 2, 39.8404],
    [1, 0, 4, 55.4065],
    [1, 1, 3, 69.3545],
    [0, 2, 4, 85.9428],
    [1, 1, 6, 95.5771],
])
"""

# 初めはh,k,l,A2obs,A2calcという順番で配置していたが、実質hklは入力しないことに気が付いた。そのため縦の順番にしてtabキーを使用できるようにした
"""
hklindexsum = []
# ピーク関数のパラメータ
for i in range(5):  # 最大6個のピーク
    hklindex = []
    # 各ガウシアンのエントリボックス (Area, Center, FWHM)
    for j in range(5):# h,k,l,A2obs,A2calcの5つ
        entry = ttk.Entry(HKL)
        entry.grid(row=1+i, column=1+j, sticky="NSEW")
        hklindex.append(entry)
        # j=0,1,2 にだけデフォルトの h, k, l を設定
        if j < 4:
            entry.insert(0, str(default_hkl_values[i][j]))
    hklindexsum.append(hklindex)

checkboxes = []
def toggle_entry_state():
    #チェックボックスの状態に応じてエントリの有効化・無効化
    for i in range(5):
        state = "normal" if checkboxes[i].get() else "readonly"
        for entry in hklindexsum[i]:
            entry.config(state=state)
"""
# hklindexsum[j][i] の構造で保存する (j=5行, i=5列)
hklindexsum = [[] for _ in range(5)]  # j=0〜4 (h, k, l, A2obs, A2calc)

# Entry配置を [j][i] に変更（列i, 行j）
for i in range(5):  # 最大5個のピーク (列方向)
    for j in range(5):  # h, k, l, A2obs, A2calc (行方向)
        entry = ttk.Entry(HKL)
        entry.grid(row=1 + j, column=1 + i, sticky="NSEW")  # ← iとjを入れ替えた
        hklindexsum[j].append(entry)
        #if i < 3:# デフォルトの値の行数によって変わる
        #    entry.insert(0, str(default_hkl_values[j][i]))  # default_hkl_values[i][j]はそのままでOK

checkboxes = []

def toggle_entry_state():
    for i in range(5):  # 各ピーク（列）
        state = "normal" if checkboxes[i].get() else "readonly"
        for j in range(5):  # 各成分（行）
            hklindexsum[i][j].config(state=state)

# チェックボックスの作成
for i in range(5):  # 最大5個のピーク
    # チェックボックス (初期状態でオフ)
    check_var = tk.BooleanVar(value=True)
    checkbox = ttk.Checkbutton(HKL, variable=check_var, command=toggle_entry_state)
    checkbox.grid(row=1 + i, column=0, sticky="NSEW")
    checkboxes.append(check_var)

# チェックボックスの初期状態を設定
toggle_entry_state()  # ここで最初に呼び出す

# A2の計算値を出力
def A2calc():
    # 入力した値を取り込む
    la=float(lc_txt1.get())
    lb=float(lc_txt2.get())
    lc=float(lc_txt3.get())
    lal=float(lc_txt4.get())
    lbe=float(lc_txt5.get())
    lga=float(lc_txt6.get())
    ei=float(mc_txt1.get())
    # ベクトルu,v,wを定義し、rluを自動で計算する
    U = [la*math.cos(math.radians(0)), 0, 0]
    V = [lb*math.cos(math.radians(lga)), lb*math.sin(math.radians(lga)), 0]
    W = [lc*math.cos(math.radians(lbe)), lc*(math.cos(math.radians(lal))-math.cos(math.radians(lbe))*math.cos(math.radians(lga)))/math.sin(math.radians(lga)), math.sqrt(lc**2-(lc*math.cos(math.radians(lbe)))**2-(lc*(math.cos(math.radians(lal))-math.cos(math.radians(lbe))*math.cos(math.radians(lga)))/math.sin(math.radians(lga)))**2)]
    astar = 2*3.141592*np.cross(V,W)/np.dot(U,np.cross(V,W))
    bstar = 2*3.141592*np.cross(W,U)/np.dot(V,np.cross(W,U))
    cstar = 2*3.141592*np.cross(U,V)/np.dot(W,np.cross(U,V))
    for i in range(5):
        if checkboxes[i].get():  # チェックボックスがオンの場合のみ
            try:
                h = float(hklindexsum[i][0].get())
                k = float(hklindexsum[i][1].get())
                l = float(hklindexsum[i][2].get())
                hkl=h*astar+k*bstar+l*cstar
                Nhkl=np.linalg.norm(hkl)
                dhkl=2*math.pi/Nhkl
                
                #lamdaとkを計算
                Li=9.045/(ei**(1/2))
                Ki=2*math.pi/Li
                tta=math.degrees(math.acos((2*Ki**2-Nhkl**2)/(2*Ki**2)))
                # A2calcの欄（5列目）に結果を挿入（既存の文字列を一度クリア）
                hklindexsum[i][4].delete(0, tk.END)
                hklindexsum[i][4].insert(0, f"{tta:.3f}")  # 小数3桁に整形
                
            except ValueError:
                pass

# A2calcボタン
calc_button = ttk.Button(HKL, text="A2 calculation",command = A2calc)
calc_button.grid(row=6, column=0, columnspan=3, sticky="NSEW")

# モノクロメータの d 値 [Å]
d_mono = 3.355/2 

# 残差関数 (最小化する目的関数)
def residuals(params, d_hkl, theta_obs):
    delta_A1, delta_A2 = params
    ei = float(mc_txt1.get())
    #lambda_ideal=9.045/(ei**(1/2))
    lambda_ideal = float(mc_txt2.get())
    #lambda_ideal = 2.372  # [Å] 理想波長(HODACA:2.372,HER:2.023)
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

# fittingボタン
def A1A2fitting():
    # メモリが解放される
    plt.clf()
    plt.close()
    
    # 入力した値を取り込む
    la=float(lc_txt1.get())
    lb=float(lc_txt2.get())
    lc=float(lc_txt3.get())
    lal=float(lc_txt4.get())
    lbe=float(lc_txt5.get())
    lga=float(lc_txt6.get())
    ei=float(mc_txt1.get())
    #lambda_ideal=9.045/(ei**(1/2))
    lambda_ideal = float(mc_txt2.get())
    #lambda_ideal = 2.372  # [Å] 理想波長(HODACA:2.372,HER:2.023)
    # ベクトルu,v,wを定義し、rluを自動で計算する
    U = [la*math.cos(math.radians(0)), 0, 0]
    V = [lb*math.cos(math.radians(lga)), lb*math.sin(math.radians(lga)), 0]
    W = [lc*math.cos(math.radians(lbe)), lc*(math.cos(math.radians(lal))-math.cos(math.radians(lbe))*math.cos(math.radians(lga)))/math.sin(math.radians(lga)), math.sqrt(lc**2-(lc*math.cos(math.radians(lbe)))**2-(lc*(math.cos(math.radians(lal))-math.cos(math.radians(lbe))*math.cos(math.radians(lga)))/math.sin(math.radians(lga)))**2)]
    astar = 2*3.141592*np.cross(V,W)/np.dot(U,np.cross(V,W))
    bstar = 2*3.141592*np.cross(W,U)/np.dot(V,np.cross(W,U))
    cstar = 2*3.141592*np.cross(U,V)/np.dot(W,np.cross(U,V))
    # h,k,l,A2obs,A2calを収納
    dataset = []
    for i in range(5):
        if checkboxes[i].get():  # チェックボックスがオンの場合のみ
            try:
                h = float(hklindexsum[i][0].get())
                k = float(hklindexsum[i][1].get())
                l = float(hklindexsum[i][2].get())
                A2obs = float(hklindexsum[i][3].get())
                
                hkl=h*astar+k*bstar+l*cstar
                Nhkl=np.linalg.norm(hkl)
                dhkl=2*math.pi/Nhkl
                
                #lamdaとkを計算
                Li=9.045/(ei**(1/2))
                Ki=2*math.pi/Li
                A2cal=math.degrees(math.acos((2*Ki**2-Nhkl**2)/(2*Ki**2)))
                # A2calcの欄（5列目）に結果を挿入（既存の文字列を一度クリア）
                hklindexsum[i][4].delete(0, tk.END)
                hklindexsum[i][4].insert(0, f"{A2cal:.3f}")  # 小数3桁に整形
                dataset.append([h,k,l,A2obs,A2cal,dhkl]) 
                
            except ValueError:
                pass
    
    H, K, L, thetaobs, thetacal,d_hkl = np.array(dataset).T
    theta_obs = np.radians(thetaobs / 2)  # 2θ → θ に変換
    theta_cal = np.radians(thetacal / 2)  # 2θ → θ に変換
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
    
    def show_fitting_result(delta_lambda, delta_theta_mono, delta_A2_fit):
        # 過去の内容を消す（毎回更新）
        for widget in RD.winfo_children():
            widget.destroy()

        # 結果文字列のリスト
        results = [
            f"wavelength offset Δλ = {delta_lambda:.6f} Å",
            f"se c1 @(c1)-({delta_theta_mono:.4f})",
            f"se a1 @(a1)-({2 * delta_theta_mono:.4f})",
            f"se a2 @(a2)-({delta_A2_fit:.4f})"
        ]

        # 行ごとにラベルとボタンを追加
        for i, line in enumerate(results):
            ttk.Label(RD, text=line).grid(row=i, column=0, sticky="NSEW")
            
            def make_copy_func(text=line):  # デフォルト引数でtextを固定
                return lambda: copy_to_clipboard(text)
            
            copy_btn = ttk.Button(RD, text="Copy left row", command=make_copy_func())
            copy_btn.grid(row=i, column=1, sticky="NSEW")

        # すべてコピーする関数
        def copy_all():
            all_text = "\n".join(results)
            copy_to_clipboard(all_text)

        # すべてコピー ボタン
        all_copy_btn = ttk.Button(RD, text="copy all", command=copy_all)
        all_copy_btn.grid(row=0, column=2, rowspan=4, sticky="NSEW")

    # 共通のクリップボードコピー関数
    def copy_to_clipboard(text):
        RD.clipboard_clear()
        RD.clipboard_append(text)
        RD.update()
    """
    # 結果表示
    def show_fitting_result(delta_lambda, delta_theta_mono, delta_A2_fit):
        # 過去の内容を消す（毎回更新）
        for widget in RD.winfo_children():
            widget.destroy()

        # 結果を表示するラベルを追加
        #ttk.Label(RD, text="フィッティング結果:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(RD, text=f"  波長のずれ Δλ = {delta_lambda:.6f} Å").grid(row=0, column=0, sticky="NSEW")
        ttk.Label(RD, text=f"  se c1 @(c1)-({delta_theta_mono:.4f})").grid(row=1, column=0, sticky="NSEW")
        ttk.Label(RD, text=f"  se a1 @(a1)-({2*delta_theta_mono:.4f})").grid(row=2, column=0, sticky="NSEW")
        ttk.Label(RD, text=f"  se a2 @(a2)-({delta_A2_fit:.4f})").grid(row=3, column=0, sticky="NSEW")
    """
    show_fitting_result(delta_lambda, delta_theta_mono, delta_A2_fit)
    """
    # 結果の表示
    print(f"フィッティング結果:")
    #print(f"  モノクロメータの回転角のずれ ΔA1 = {delta_A1_fit:.4f}°")
    #print(f"  2θ のオフセット ΔA2 = {delta_A2_fit:.4f}°")
    print(f"  波長のずれ Δλ = {delta_lambda:.6f} Å")
    #print(f"  モノクロメータの回転角のずれ Δθ_mono = {delta_theta_mono:.4f}°")

    print(f"  se c1 @(c1)-({delta_theta_mono:.4f})")
    print(f"  se a1 @(a1)-({2*delta_theta_mono:.4f})")
    print(f"  se a2 @(a2)-({(delta_A2_fit):.4f})")
    """
    # プロット
    theta_fit = np.arcsin(lambda_fit / (2 * d_hkl)) + np.radians(delta_A2_fit / 2)
    plt.scatter(np.degrees(theta_obs * 2), np.degrees(theta_fit * 2), label="Fitted", color="red")
    plt.plot([30, 110], [30, 110], linestyle="--", color="black", label="y=x (ideal)")
    plt.xlabel("Observed 2θ [deg]")
    plt.ylabel("Fitted 2θ [deg]")
    plt.legend()
    plt.grid()
    plt.show()
    
# fittingボタン
fit_button = ttk.Button(HKL, text="fitting",command = A1A2fitting)
fit_button.grid(row=6, column=3, columnspan=3, sticky="NSEW")

# 1. 結果表示用のフレームを作成（初回のみでOK）
result_frame = ttk.Frame(root)  # root はあなたのメインウィンドウ
result_frame.grid(row=99, column=0, columnspan=10, sticky="NSEW", padx=5, pady=5)

#メニューバーの作成
menubar = tk.Menu(root)
root.configure(menu=menubar)

# iniファイルの読み込み
def load_values_from_ini():
    config = configparser.ConfigParser()
    # .exe化した場合に対応する
    if getattr(sys, 'frozen', False):
        # .exeの場合、sys.argv[0]が実行ファイルのパスになる
        ini_path = os.path.join(os.path.dirname(sys.argv[0]), 'config.ini')
    else:
        # .pyの場合、__file__がスクリプトのパスになる
        ini_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(ini_path)
    
    # 各エントリに対応する値を読み込み、挿入
    lc_txt1.delete(0, tk.END)  # 既存の値をクリア
    lc_txt1.insert(0, config['LC'].get('a', '4.758'))
    lc_txt2.delete(0, tk.END)  # 既存の値をクリア
    lc_txt2.insert(0, config['LC'].get('b', '4.758'))
    lc_txt3.delete(0, tk.END)  # 既存の値をクリア
    lc_txt3.insert(0, config['LC'].get('c', '12.991'))
    lc_txt4.delete(0, tk.END)  # 既存の値をクリア
    lc_txt4.insert(0, config['LC'].get('alpha', '90'))
    lc_txt5.delete(0, tk.END)  # 既存の値をクリア
    lc_txt5.insert(0, config['LC'].get('beta', '90'))
    lc_txt6.delete(0, tk.END)  # 既存の値をクリア
    lc_txt6.insert(0, config['LC'].get('gamma', '120'))
    
    mc_txt1.delete(0, tk.END)  # 既存の値をクリア
    mc_txt1.insert(0, config['MC'].get('energy', '14.5404'))
    mc_txt2.delete(0, tk.END)  # 既存の値をクリア
    mc_txt2.insert(0, config['MC'].get('lambda', '2.3720'))
    mc_txt3.delete(0, tk.END)  # 既存の値をクリア
    mc_txt3.insert(0, config['MC'].get('wavelength', '2.6489'))
    
    # MLセクション取得
    ml_section = config["ML"]
    # Entryに値を挿入（h, k, l のみ）
    for i in range(5):  # ピーク番号
        for j in range(3):  # h, k, l に対応
            key = f"{['h', 'k', 'l'][j]}{i+1}"  # e.g., h1, k1, l1,...
            value = ml_section.get(key, "")  # デフォルト空文字
            hklindexsum[i][j].delete(0, tk.END)     # 既存の内容を削除
            hklindexsum[i][j].insert(0, value)
def save_values_to_ini():
    """
    現在のウィジェットの値をINIファイルに保存する
    """
    config = configparser.ConfigParser()
    
    # INIファイルのパスを決定
    if getattr(sys, 'frozen', False):
        ini_path = os.path.join(os.path.dirname(sys.argv[0]), 'config.ini')
    else:
        ini_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    
    # 既存のINIファイルを読み込む
    if os.path.exists(ini_path):
        config.read(ini_path, encoding='utf-8')  # UTF-8で読み込み
    
    # LC セクション（格子定数）
    config["LC"] = {
        "a": lc_txt1.get(),
        "b": lc_txt2.get(),
        "c": lc_txt3.get(),
        "alpha": lc_txt4.get(),
        "beta": lc_txt5.get(),
        "gamma": lc_txt6.get(),
    }

    # MC セクション（モノクロメータ）
    config["MC"] = {
        "energy": mc_txt1.get(),
        "lambda": mc_txt2.get(),
        "wavelength": mc_txt3.get(),
    }

    # ML セクション（hklリスト）
    ml_section = {}
    for i in range(5):  # ピーク番号（列方向）
        for j, axis in enumerate(['h', 'k', 'l']):  # h, k, l に対応（行方向）
            value = hklindexsum[i][j].get()
            key = f"{axis}{i+1}"  # 例: h1, k1, l1 ...
            ml_section[key] = value
    config["ML"] = ml_section
    
    # 保存処理
    with open(ini_path, 'w') as configfile:
        config.write(configfile)

# アプリ起動時にデフォルト値を読み込む
load_values_from_ini()

#fileメニュー(setting)
filemenu = tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label="ini.file",menu=filemenu)
#fileメニューにini fileのload
filemenu.add_command(label="load ini.file",command=load_values_from_ini)
#fileメニューにexitを追加。ついでにexit funcも実装

#fileメニューにini fileのsave
filemenu.add_command(label="save ini.file",command=save_values_to_ini)

#window状態の維持
root.mainloop()

#############
# pyinstaller 
# 最初にディレクトリ移動
# C:\DATA_HK\python\HODACA_calibration
# pyinstaller -F --noconsole calibrationGUI.py
