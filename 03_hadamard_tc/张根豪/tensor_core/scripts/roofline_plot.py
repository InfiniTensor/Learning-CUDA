"""Shared measured-DRAM roofline rendering; no GPU or plotting imports at import time."""


def plot_roofline(data, output, dram_peak_gbs, fp32_peak_tflops):
    if not data or len({row['gpu'] for row in data}) != 1:
        raise ValueError('roofline requires nonempty data from exactly one GPU model')
    if dram_peak_gbs <= 0 or fp32_peak_tflops <= 0:
        raise ValueError('roofline peaks must be positive')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, policy in zip(axes, ('all', 'none')):
        x = np.logspace(-2, 4, 300)
        ax.loglog(x, np.minimum(x * dram_peak_gbs, fp32_peak_tflops * 1000),
                  'k--', label='DRAM + non-TC FP32 reference')
        for color, mode in enumerate(('warp', 'wmma_fast', 'wmma_split', 'ptx_fast', 'ptx_split')):
            rows = [row for row in data if row['backend'] == mode
                    and row['cache_control'] == policy and row['useful_flops_per_dram_byte']]
            for index, row in enumerate(rows):
                ax.scatter(row['useful_flops_per_dram_byte'], row['useful_gflops'],
                           color=f'C{color}', marker={64: 'o', 128: 's', 256: '^'}[row['dim']],
                           label=mode if index == 0 else None)
        intensities = [row['useful_flops_per_dram_byte'] for row in data
                       if row['cache_control'] == policy and row['useful_flops_per_dram_byte']]
        if not intensities:
            plt.close(fig)
            raise ValueError(f'no positive DRAM intensity for cache-control={policy}')
        ax.set_xlim(min(intensities) / 2,
                    max(max(intensities) * 2, fp32_peak_tflops * 1000 / dram_peak_gbs * 2))
        ax.set(title=f"{data[0]['gpu']}: cache-control={policy}",
               xlabel='Useful FHT FLOPs / measured DRAM byte', ylabel='Useful FHT GFLOP/s')
        ax.grid(alpha=.2)
        legend = ax.legend(fontsize=7, loc='upper left')
        ax.add_artist(legend)
        ax.legend(handles=[Line2D([], [], color='black', marker=marker, linestyle='', label=f'd={dim}')
                           for dim, marker in ((64, 'o'), (128, 's'), (256, '^'))],
                  loc='lower right', fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
