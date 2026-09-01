import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_PATH = (
    "/home/serenaliu/caltech_linc_home/open-ant/agents/sac/runs/no_adam/idbd_largerweight_layernorm/sac_sim1_20260831-145506_seed_3/weights.pth"
)
LR_STEP = 40000
OUTPUT_DIR = os.path.join(
    os.path.dirname(CHECKPOINT_PATH),
    "weight_heatmaps",
)

OLD_HIDDEN = 256


# ============================================================
# HELPERS
# ============================================================

def plot_linear_layer(
    state_dict,
    weight_name,
    bias_name,
    title,
    output_path,
    old_hidden=256,
):
    """
    Plot:
        left  = weight matrix
        right = bias vector

    PyTorch Linear weight shape:
        [out_features, in_features]

    Therefore:
        rows    = neurons in current layer
        columns = neurons/features feeding into layer
    """

    weight = (
        state_dict[weight_name]
        .detach()
        .cpu()
        .numpy()
    )
    abs_w = np.abs(weight)

    print(
        f"{title}: "
        f"shape={weight.shape}, "
        f"mean|w|={abs_w.mean():.6g}, "
        f"median|w|={np.median(abs_w):.6g}, "
        f"p95={np.percentile(abs_w, 95):.6g}, "
        f"p99={np.percentile(abs_w, 99):.6g}, "
        f"max|w|={abs_w.max():.6g}"
    )

    bias = (
        state_dict[bias_name]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    # weight[2, :] = 1.5   # Output neuron 2 has strong positive weights across all inputs
    # weight[5, 3:6] = -1.8 # Output neuron 5 has a block of strong negative weights 


    # Use same symmetric range around zero for easy comparison.
    max_abs = np.max(np.abs(weight))

    if max_abs == 0:
        max_abs = 1e-12

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={
            "width_ratios": [12, 1]
        },
    )

    # ========================================================
    # WEIGHT MATRIX
    # ========================================================

    im = axes[0].imshow(
        weight,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    axes[0].set_title(
        f"{title}: Weights"
    )

    axes[0].set_xlabel(
        "Input Features / Previous-Layer Neurons"
    )

    axes[0].set_ylabel(
        "Output Neurons"
    )

    fig.colorbar(
        im,
        ax=axes[0],
        label="Weight Value",
    )


    # ========================================================
    # DRAW OLD / NEW NEURON BOUNDARY
    # ========================================================

    rows, cols = weight.shape

    # Horizontal line:
    # separates old output neurons from new output neurons.
    if rows > old_hidden:
        axes[0].axhline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )

    # Vertical line:
    # separates old input neurons from new input neurons.
    #
    # Only relevant when the input dimension itself
    # contains the expanded hidden layer.
    if cols > old_hidden:
        axes[0].axvline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )


    # ========================================================
    # BIAS VECTOR
    # ========================================================

    bias_max = np.max(np.abs(bias))

    if bias_max == 0:
        bias_max = 1e-12

    axes[1].imshow(
        bias,
        aspect="auto",
        cmap="coolwarm",
        vmin=-bias_max,
        vmax=bias_max,
    )

    axes[1].set_title(
        "Bias"
    )

    axes[1].set_xticks([])

    axes[1].set_ylabel(
        "Output Neurons"
    )

    if len(bias) > old_hidden:
        axes[1].axhline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_learning_rate_layer(
    lr_data,
    weight_name,
    bias_name,
    title,
    output_path,
    old_hidden=256,
):
    """
    Plot individual per-parameter learning rates.

    left:
        learning rates for weight matrix

    right:
        learning rates for bias vector

    Learning rates are plotted as log10(LR),
    because IDBD learning rates can span many orders of magnitude.
    """

    weight_lr = np.asarray(
        lr_data[weight_name]
    )

    bias_lr = (
        np.asarray(
            lr_data[bias_name]
        )
        .reshape(-1, 1)
    )


    # ========================================================
    # PRINT LR STATS
    # ========================================================

    print(
        f"{title} LR: "
        f"shape={weight_lr.shape}, "
        f"mean={weight_lr.mean():.6g}, "
        f"median={np.median(weight_lr):.6g}, "
        f"p95={np.percentile(weight_lr, 95):.6g}, "
        f"p99={np.percentile(weight_lr, 99):.6g}, "
        f"max={weight_lr.max():.6g}"
    )


    # ========================================================
    # LOG10 LEARNING RATE
    # ========================================================

    # Prevent log10(0) if an LR underflowed to exactly zero.
    eps = 1e-30

    log_weight_lr = np.log10(
        np.maximum(weight_lr, eps)
    )

    log_bias_lr = np.log10(
        np.maximum(bias_lr, eps)
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={
            "width_ratios": [12, 1]
        },
    )


    # ========================================================
    # WEIGHT LEARNING RATES
    # ========================================================

    im = axes[0].imshow(
        log_weight_lr,
        aspect="auto",
        cmap="viridis",
    )

    axes[0].set_title(
        f"{title}: Per-Weight Learning Rate"
    )

    axes[0].set_xlabel(
        "Input Features / Previous-Layer Neurons"
    )

    axes[0].set_ylabel(
        "Output Neurons"
    )

    fig.colorbar(
        im,
        ax=axes[0],
        label="log10(Learning Rate)",
    )


    # ========================================================
    # OLD / NEW NEURON BOUNDARY
    # ========================================================

    rows, cols = weight_lr.shape

    if rows > old_hidden:
        axes[0].axhline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )

    if cols > old_hidden:
        axes[0].axvline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )


    # ========================================================
    # BIAS LEARNING RATES
    # ========================================================

    axes[1].imshow(
        log_bias_lr,
        aspect="auto",
        cmap="viridis",
    )

    axes[1].set_title(
        "Bias LR"
    )

    axes[1].set_xticks([])

    axes[1].set_ylabel(
        "Output Neurons"
    )

    if len(bias_lr) > old_hidden:
        axes[1].axhline(
            old_hidden - 0.5,
            linestyle="--",
            linewidth=1.5,
        )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
# ============================================================
# LOAD CHECKPOINT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print(
    "Checkpoint keys:",
    checkpoint.keys(),
)

# ============================================================
# LOAD MATCHING PER-WEIGHT LEARNING RATES
# ============================================================

LR_PATH = os.path.join(
    os.path.dirname(CHECKPOINT_PATH),
    "learning_rates",
    f"lr_step_{LR_STEP}.npz",
)

print(f"Loading learning rates from:\n{LR_PATH}")

lr_data = np.load(LR_PATH)

print("\nLearning-rate keys:")
for key in lr_data.files:
    print("   ", key)
# ============================================================
# ACTOR
# ============================================================

actor = checkpoint["actor"]

plot_linear_layer(
    actor,
    "fc1.weight",
    "fc1.bias",
    "Actor FC1",
    os.path.join(
        OUTPUT_DIR,
        "actor_fc1.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    actor,
    "fc2.weight",
    "fc2.bias",
    "Actor FC2",
    os.path.join(
        OUTPUT_DIR,
        "actor_fc2.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    actor,
    "fc_mean.weight",
    "fc_mean.bias",
    "Actor Mean Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_mean.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    actor,
    "fc_logstd.weight",
    "fc_logstd.bias",
    "Actor Log-Std Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_logstd.png",
    ),
    OLD_HIDDEN,
)


# ============================================================
# QF1
# ============================================================

qf1 = checkpoint["qf1"]

plot_linear_layer(
    qf1,
    "fc1.weight",
    "fc1.bias",
    "QF1 FC1",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc1.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    qf1,
    "fc2.weight",
    "fc2.bias",
    "QF1 FC2",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc2.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    qf1,
    "fc3.weight",
    "fc3.bias",
    "QF1 Output",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc3.png",
    ),
    OLD_HIDDEN,
)


# ============================================================
# QF2
# ============================================================

qf2 = checkpoint["qf2"]

plot_linear_layer(
    qf2,
    "fc1.weight",
    "fc1.bias",
    "QF2 FC1",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc1.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    qf2,
    "fc2.weight",
    "fc2.bias",
    "QF2 FC2",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc2.png",
    ),
    OLD_HIDDEN,
)

plot_linear_layer(
    qf2,
    "fc3.weight",
    "fc3.bias",
    "QF2 Output",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc3.png",
    ),
    OLD_HIDDEN,
)
# ============================================================
# ACTOR LEARNING-RATE HEATMAPS
# ============================================================

plot_learning_rate_layer(
    lr_data,
    "actor.fc1.weight",
    "actor.fc1.bias",
    "Actor FC1",
    os.path.join(
        OUTPUT_DIR,
        "actor_fc1_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "actor.fc2.weight",
    "actor.fc2.bias",
    "Actor FC2",
    os.path.join(
        OUTPUT_DIR,
        "actor_fc2_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "actor.fc_mean.weight",
    "actor.fc_mean.bias",
    "Actor Mean Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_mean_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "actor.fc_logstd.weight",
    "actor.fc_logstd.bias",
    "Actor Log-Std Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_logstd_lr.png",
    ),
    OLD_HIDDEN,
)
# ============================================================
# QF1 LEARNING-RATE HEATMAPS
# ============================================================

plot_learning_rate_layer(
    lr_data,
    "qf1.fc1.weight",
    "qf1.fc1.bias",
    "QF1 FC1",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc1_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "qf1.fc2.weight",
    "qf1.fc2.bias",
    "QF1 FC2",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc2_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "qf1.fc3.weight",
    "qf1.fc3.bias",
    "QF1 Output",
    os.path.join(
        OUTPUT_DIR,
        "qf1_fc3_lr.png",
    ),
    OLD_HIDDEN,
)
# ============================================================
# QF2 LEARNING-RATE HEATMAPS
# ============================================================

plot_learning_rate_layer(
    lr_data,
    "qf2.fc1.weight",
    "qf2.fc1.bias",
    "QF2 FC1",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc1_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "qf2.fc2.weight",
    "qf2.fc2.bias",
    "QF2 FC2",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc2_lr.png",
    ),
    OLD_HIDDEN,
)

plot_learning_rate_layer(
    lr_data,
    "qf2.fc3.weight",
    "qf2.fc3.bias",
    "QF2 Output",
    os.path.join(
        OUTPUT_DIR,
        "qf2_fc3_lr.png",
    ),
    OLD_HIDDEN,
)
print(
    f"\nSaved heatmaps to:\n{OUTPUT_DIR}"
)
