import pandas as pd
import matplotlib.pyplot as plt

def plot_dt(log_file: str, output_file: str):
    df = pd.read_csv(log_file)
    # df = df[df['dt_step'] <= ]
    plt.figure(figsize=(10, 6))
    plt.scatter(df['step'][1:], df['dt_step'][1:], label='dt_step', s=0.5)
    plt.xlabel('Step')
    plt.ylabel('Time (seconds)')
    plt.title('Step Time per Step')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    log_file = "agents/mpo/runs/retrace_20260605-135051_seed_0/performance_variables.csv"  # Replace with your actual log file path
    output_file = "step_time_plot.png"  # Desired output file name
    plot_dt(log_file, output_file)