"""Plot MPO sim pre-training → HW continual learning for seeds 6 and 3.

Two panels (one per seed). Each shows the sim phase followed by the HW
continual-learning phase on a continuous time axis, separated by a vertical line.

Usage (from repo root or utils/):
    python3 utils/plot_transfer.py [--output transfer.pdf] [--smooth 300] [--dt 0.12]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams.update({'font.size': 13})

COLORS = {
    'sim_6': '#4477AA',  # steel blue  — seed 6 sim
    'hw_6':  '#228833',  # forest green — seed 6 HW
    'sim_3': '#EE6677',  # coral red   — seed 3 sim
    'hw_3':  '#AA3377',  # purple      — seed 3 HW
}

HERE      = os.path.dirname(os.path.abspath(__file__))
MPO_SIM   = os.path.join(HERE, '..', 'agents', 'mpo', 'runs')
MPO_HW    = os.path.join(HERE, '..', 'agents', 'mpo', 'runs_hw')

RUNS = {
    6: {
        'sim': os.path.join(MPO_SIM, 'initial_sim_train_20260611-123528_seed_6'),
        'hw':  os.path.join(MPO_HW,  'continual_learning_from_seed_6_20260611-164544_seed_1'),
    },
    3: {
        'sim': os.path.join(MPO_SIM, 'initial_sim_train_20260611-113058_seed_3'),
        'hw':  os.path.join(MPO_HW,  'continual_learning_from_seed_3_20260611-175533_seed_1'),
    },
}

COL = 'average_reward_per_second'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, default='transfer_mpo_hw.pdf')
    p.add_argument('--smooth', type=int, default=300,
                   help='Rolling-mean window (rows ≈ 1 per step).')
    p.add_argument('--dt', type=float, default=0.12,
                   help='Timestep in seconds (same for sim and HW).')
    return p.parse_args()


def load(run_dir):
    path = os.path.join(run_dir, 'performance_variables.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Not found: {path}')
    return pd.read_csv(path)


def smooth(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def plot_seed(ax, seed, args):
    paths = RUNS[seed]
    df_sim = load(paths['sim']).sort_values('step')
    df_hw  = load(paths['hw']).sort_values('step')

    dt_per_min = args.dt / 60.0

    # Sim time axis (minutes of simulated experience).
    df_sim['t'] = df_sim['step'] * dt_per_min
    sim_end_t   = df_sim['t'].max()

    # HW time axis: offset so it continues from sim end.
    hw_step0    = df_hw['step'].min()
    df_hw['t']  = sim_end_t + (df_hw['step'] - hw_step0) * dt_per_min

    c_sim = COLORS[f'sim_{seed}']
    c_hw  = COLORS[f'hw_{seed}']

    ax.plot(df_sim['t'].values, smooth(df_sim[COL], args.smooth).values,
            color=c_sim, linewidth=1.8, label='MPO — sim pre-training')
    ax.plot(df_hw['t'].values, smooth(df_hw[COL], args.smooth).values,
            color=c_hw, linewidth=1.8, label='MPO — HW continual learning')

    ax.axvline(x=sim_end_t, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
               label=f'Sim → HW ({sim_end_t:.1f} min)')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Experience (simulated minutes)')
    ax.set_ylabel('Avg. reward / s')
    ax.set_title(f'Seed {seed}: MPO sim → HW continual learning')
    ax.legend(loc='upper left', fontsize=10)


def main():
    args = parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, seed in zip(axes, [6, 3]):
        plot_seed(ax, seed, args)

    fig.suptitle('MPO: Sim Pre-training → Hardware Continual Learning', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'[√] Saved to {args.output}')
    plt.show()


if __name__ == '__main__':
    main()
