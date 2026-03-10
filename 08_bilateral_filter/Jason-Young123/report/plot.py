import matplotlib.pyplot as plt
import numpy as np

def plot_curve(speedup_gray, speedup_rgb):
    radius = list(range(1, 11))
    
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(radius, speedup_gray, marker='o', markersize=6, linewidth=2, 
            color='gray', label='Gray')
    ax.plot(radius, speedup_rgb, marker='s', markersize=6, linewidth=2, 
            color='red', label='RGB')
    

    fig.text(0.5, 0.95, 'Speedup Analysis (Platform: Nvidia)', 
             ha='center', va='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.90, 'GPU: A100  /  CPU: Intel(R) Xeon(R) Processor @ 2.90GHz', 
             ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('Radius', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup x', fontsize=13, fontweight='bold')
    ax.set_xticks(radius)
    ax.grid(True, linestyle=':', alpha=0.6)

    #ax.set_ylim(0, 34)
    
    ax.legend(loc='upper center', ncol=2, frameon=True, shadow=True)
    
    plt.subplots_adjust(top=0.85)

    #plt.savefig('speedup_nvidia.png', dpi=300, bbox_inches='tight')
    plt.show()






def plot_bar(thruput_gray, thruput_rgb):
    radius = np.arange(1, 11)  # 横坐标 1-10
    width = 0.35              # 柱子的宽度
    
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制并排柱状图
    ax.bar(radius - width/2, thruput_gray, width, color='gray', label='Gray Throughput')
    ax.bar(radius + width/2, thruput_rgb, width, color='lightskyblue', label='RGB Throughput')
    
    # --- 关键修改：红色虚线，不带 label (不会出现在图例中) ---
    fps_4k_60 = 497.66
    ax.axhline(y=fps_4k_60, color='red', linestyle='--', linewidth=1.5)

    # 标注文字保持红色以匹配线条
    ax.text(11, fps_4k_60 + 60, '4K 60fps', 
            color='red', fontweight='bold', fontsize=13, va='bottom', ha = 'right')

    # --- 标题设置 ---
    fig.text(0.5, 0.95, 'Throughput Analysis (Platform: Iluvatar)', 
             ha='center', va='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.90, 'GPU: BI100', 
             ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('Radius', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (MP/s)', fontsize=13, fontweight='bold')
    ax.set_xticks(radius)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # 设置 Y 轴范围
    y_max = max(max(thruput_gray), max(thruput_rgb))
    ax.set_ylim(0, y_max * 1.05)
    
    # 图例设置：此时只会显示 Gray 和 RGB 两项
    ax.legend(loc='upper center', ncol=2, frameon=True, shadow=True, fontsize=11)
    
    plt.subplots_adjust(top=0.85)
    #plt.savefig('throughput_iluvatar.png', dpi=300, bbox_inches='tight')
    plt.show()




def plot_bar_comparison(nvidia, moore, metax, iluvatar):
    # 横坐标为 2, 4, 6, 8, 10
    radius_labels = [2, 4, 6, 8, 10]
    x = np.arange(len(radius_labels))  # 标签位置
    width = 0.2  # 每个柱子的宽度
    
    # 设置字体和基础样式
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # 选取四个不同深度的蓝色
    colors = ['#B3E5FC', '#81D4FA', '#039BE5', '#01579B']

    # 绘制四个平台的并列柱状图
    ax.bar(x - 1.5*width, nvidia, width, color=colors[0], label='Nvidia(A100)')
    ax.bar(x - 0.5*width, moore, width, color=colors[1], label='Moore(S5000)')
    ax.bar(x + 0.5*width, metax, width, color=colors[2], label='Metax(C500)')
    ax.bar(x + 1.5*width, iluvatar, width, color=colors[3], label='Iluvatar(BI100)')
    
    # 绘制 4K 60fps 参考基准线
    fps_4k_60 = 497.66
    ax.axhline(y=fps_4k_60, color='red', linestyle='--', linewidth=1.5, zorder=3)
    ax.text(len(x)-0.5, fps_4k_60 + 50, '4K 60fps', 
            color='red', fontweight='bold', fontsize=11, va='bottom', ha='right')

    # 设置标题和标签
    fig.text(0.5, 0.94, 'Throughput Comparison among 4 Platforms', 
             ha='center', va='center', fontsize=18, fontweight='bold')
    
    ax.set_xlabel('Radius', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (MP/s)', fontsize=13, fontweight='bold')
    
    # 设置横坐标刻度
    ax.set_xticks(x)
    ax.set_xticklabels(radius_labels)
    
    # 辅助线和背景
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # 设置 Y 轴范围（自适应并留出图例空间）
    y_all = nvidia + moore + metax + iluvatar
    ax.set_ylim(0, max(y_all) * 1.1)
    
    # 图例设置：ncol=4 放在正上方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              ncol=4, frameon=True, shadow=True, fontsize=10)
    
    plt.subplots_adjust(top=0.88)
    plt.savefig('./pic/throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()



speedup_gray_nvidia = [4.27553, 3.99363, 3.93609, 4.14105, 3.92931, 3.99847, 3.02489, 4.03044, 2.73277, 2.94585]
speedup_rgb_nvidia = [14.1611, 8.02206, 9.37769, 11.2139, 10.7873, 9.27599, 13.915, 16.017, 16.684, 17.9033]

thruput_gray_nvidia = [3254.41, 3060.33, 2763.45, 2478.31, 2130.61, 1878.15, 1079.03, 972.94, 690.79, 632.35]
thruput_rgb_nvidia = [1650.54, 1582.13, 1468.59, 1354.47, 1201.05, 806.97, 742.75, 551.99, 442.79, 405.94]

speedup_gray_moore = [3.61347, 4.35318, 9.09573, 10.029, 10.792, 10.9914, 11.4681, 11.5015, 11.7973, 11.8315]
speedup_rgb_moore = [16.8466, 18.1259, 27.2701, 28.9265, 29.0732, 29.556, 29.8733, 30.0655, 30.1244, 30.1175]

thruput_gray_moore = [12522.78, 9385.87, 6173.55, 4329.85, 2957.89, 2239.56, 1759.93, 1371.34, 1091.28, 885.31]
thruput_rgb_moore = [4833.82, 3833.93, 2739.49, 2022.33, 1424.69, 1104.50, 880.56, 694.45, 556.19, 453.54]

speedup_gray_metax = [1.98467, 1.79002, 2.67978, 2.62693, 2.80345, 2.69172, 2.73795, 2.69132, 2.73793, 2.72632]
speedup_rgb_metax = [21.1496, 15.6313, 12.5088, 10.5937, 8.23794, 7.87266, 4.59177, 4.41521, 4.16321, 5.26927]

thruput_gray_metax = [9551.09, 5931.08, 3457.88, 2271.39, 1494.44, 1102.80, 862.00, 664.94, 527.62, 425.89]
thruput_rgb_metax = [4361.43, 3215.41, 2188.33, 1576.27, 1086.52, 829.44, 655.42, 506.24, 407.79, 330.19]


speedup_gray_iluvatar = [8.35226, 7.02025, 6.36762, 6.03288, 4.52927, 3.66657, 2.98139, 2.46526, 2.38591, 2.20986]
speedup_rgb_iluvatar = [11.4367, 10.1008, 9.04011, 8.06227, 6.37318, 5.15521, 4.31887, 3.92115, 3.71837, 3.51772]

thruput_gray_iluvatar = [8594.56, 6140.55, 3894.23, 2685.26, 1799.97, 1350.31, 1055.55, 817.68, 647.83, 523.72]
thruput_rgb_iluvatar = [2662.75, 2254.44, 1660.98, 1256.35, 901.42, 701.81, 562.32, 444.78, 357.13, 291.72]



thruput_rgb_nvidia1 = [1582.13, 1354.47, 806.97, 551.99, 405.94]
thruput_rgb_moore1 = [3833.93, 2022.33, 1104.50, 694.45, 453.54]
thruput_rgb_metax1 = [3215.41, 1576.27, 829.44, 506.24, 330.19]
thruput_rgb_iluvatar1 = [2254.44, 1256.35, 701.81, 444.78, 291.72]



if __name__ == "__main__":
    #plot_curve(speedup_gray_nvidia, speedup_rgb_nvidia)
    #plot_bar(thruput_gray_iluvatar, thruput_rgb_iluvatar)
    #plot_bar(thruput_gray_nvidia, thruput_rgb_nvidia)

    plot_bar_comparison(thruput_rgb_nvidia1, thruput_rgb_moore1, thruput_rgb_metax1, thruput_rgb_iluvatar1)