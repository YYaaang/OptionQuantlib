import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt

def plot_fig(data):
    # 1. 创建画布（调整大小适应大量曲线）
    plt.figure(figsize=(15, 8), dpi=100)

    # 2. 绘制所有折线（优化显示效果）
    for i, arr in enumerate(data):
        plt.plot(arr,
                 linewidth=0.9,  # 细线避免重叠
                 alpha=0.3,  # 半透明区分重叠曲线
                 # color='blue'  # 统一颜色
                 )

        # 3. 添加图表元素
    plt.title(f"Curves Visualization", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)  # 辅助网格线

    # 4. 显示/保存
    plt.tight_layout()  # 自动调整间距
    # plt.savefig('massive_lines.png', bbox_inches='tight')  # 保存高清图片
    plt.show()

def check_list_length(*args):
    vars = [i for i in args if isinstance(i, list) or isinstance(i, np.ndarray)]

    if len(vars) == 0 :
        return [[i] for i in args]

    lengths = [len(i) for i in vars]

    if len(set(lengths)) > 1:
        raise ('wrong input length')

    l = lengths[0]

    return [i if isinstance(i, list) or isinstance(i, np.ndarray) else [i] * l for i in args]

def set_option_name(stock_code, date, strike, option_type):
    if option_type == -1:
        option_type = 'P'
    else:
        option_type = 'C'
    return (f'{stock_code}'
            f'{str(date.year())[-2:]}'
            f'{str(date.month()).zfill(2)}'
            f'{str(date.dayOfMonth()).zfill(2)}'
            f'{option_type}'
            f'{str(int(strike * 1000)).zfill(8)}')

def set_analytic_engine(process_type, process):
    if process_type == 'heston':
        return ql.AnalyticHestonEngine(ql.HestonModel(process))
    else:
        return ql.AnalyticEuropeanEngine(process)

def set_fd_engine(process_type,
                  process,
                  tGrid = 100,
                  xGrid = 100,
                  vGrid = 50):
    if process_type == 'heston':
        return ql.FdHestonVanillaEngine(
            ql.HestonModel(process),
            tGrid,
            xGrid,
            vGrid
        )
    else:
        return ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)

