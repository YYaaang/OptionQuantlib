import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt

def plot_multi_y_axis(x_data, y_data, title=None, labels=None, colors=None, linestyles=None, figsize=(10, 6)):
    """
    绘制多 Y 轴折线图，每个数据系列有独立的 Y 轴

    参数:
        x_data : array-like
            X 轴数据（一维数组或列名，若为 DataFrame）
        y_data : 2D array-like 或 DataFrame
            Y 轴数据（二维数组或 DataFrame 的列）
        labels : list, optional
            数据系列名称（用于图例）
        colors : list, optional
            各线条颜色（默认使用 Matplotlib 颜色循环）
        linestyles : list, optional
            各线条样式（如 '-', '--', ':'）
        figsize : tuple, optional
            图表大小（默认 (10, 6)）
    """
    # 数据预处理
    if isinstance(y_data, pd.DataFrame):
        y_arrays = y_data.values  # 转换为二维数组
        if labels is None:
            labels = y_data.columns.tolist()
    else:
        y_arrays = np.atleast_2d(y_data)
        if y_arrays.shape[0] == 1:  # 若输入为一维数组，转置为二维
            y_arrays = y_arrays.T

    n_lines = y_arrays.shape[1] if y_arrays.ndim == 2 else 1
    if labels is None:
        labels = [f'Series {i + 1}' for i in range(n_lines)]
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_lines]
    if linestyles is None:
        linestyles = ['-'] * n_lines

    # 创建图表和主 Y 轴
    fig, ax1 = plt.subplots(figsize=figsize)
    axes = [ax1]
    lines = []

    # 绘制第一条线（主 Y 轴）
    lines.append(ax1.plot(x_data, y_arrays[:, 0], label=labels[0],
                          color=colors[0], linestyle=linestyles[0])[0])
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel(labels[0], color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])

    # 动态调整右侧空白
    right_margin = 0.7 - 0.1 * (n_lines - 2)  # 根据轴数量调整
    fig.subplots_adjust(right=right_margin)

    # 添加额外 Y 轴并绘制剩余线条
    for i in range(1, n_lines):
        ax = ax1.twinx()
        ax.spines['right'].set_position(('outward', 60 * i))  # 轴间距
        lines.append(ax.plot(x_data, y_arrays[:, i], label=labels[i],
                             color=colors[i], linestyle=linestyles[i])[0])
        ax.set_ylabel(labels[i], color=colors[i])
        ax.tick_params(axis='y', labelcolor=colors[i])
        axes.append(ax)

    # 合并图例
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.05, 0.95))

    if title is None:
        title = 'Multi Y Axis'
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    return

def unix_time_to_ql_dates(all_dates: np.array):
    return [ql.Date(int(i)) for i in all_dates // 86400 + 25569]

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

def set_mc_engine(process_type,
                  process,
                  steps=2,
                  rng="pseudorandom",  # could use "lowdiscrepancy"
                  numPaths=100000,
                  ):

    if process_type == 'heston':
        return ql.MCEuropeanHestonEngine(process, rng, steps, requiredSamples=numPaths)
    else:
        return ql.MCEuropeanEngine(process, rng, steps, requiredSamples=numPaths)

if __name__ == '__main__':
    # 生成示例数据
    x = np.linspace(0, 10, 100)
    y = np.column_stack([np.sin(x), np.cos(x), x ** 2])

    # 调用函数
    plot_multi_y_axis(x, y, labels=['sin(x)', 'cos(x)', 'x^2'],
                      colors=['red', 'blue', 'green'],
                      linestyles=['-', '--', ':'])

    # 生成 DataFrame
    df = pd.DataFrame({
        'x': np.arange(1, 11),
        'Price': np.random.rand(10) * 100,
        'Volume': np.random.rand(10) * 1000,
        'Volatility': np.random.rand(10)
    })

    # 调用函数（指定 x 列和 y 列）
    plot_multi_y_axis(df['x'], df[['Price', 'Volume', 'Volatility']],
                      figsize=(12, 5))