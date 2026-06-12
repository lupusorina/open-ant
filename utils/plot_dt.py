import pandas as pd
import matplotlib.pyplot as plt

def plot_dt(log_file: str, output_file: str):
    df = pd.read_csv(log_file)
    df = df[df['dt_step'] <= 10]
    even = df[df['step'] % 2 == 0]
    odd = df[df['step'] % 2 == 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(even['step'][1:], even['dt_step'][1:], label='dt_step', s=0.5, color='blue', alpha=0.5)
    plt.scatter(odd['step'][1:], odd['dt_step'][1:], label='dt_step', s=0.5, color='red', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Time (seconds)')
    plt.title('Step Time per Step')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    log_file = "agents/mpo/runs_hw/baseline_from_scratch_20260605-154223_seed_1/performance_variables.csv"  # Replace with your actual log file path
    # log_file = "agents/sac/runs_sim_test/trial_1_20260608-101109_seed_1/info_sac_logs.csv"
    output_file = "step_time_plot.png"  # Desired output file name
    plot_dt(log_file, output_file)