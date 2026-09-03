import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_PATH = (
    "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/2mpo/continuous_mpo_20260810-191808_seed_2/weights_and_args/checkpoint_200000.pth"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(CHECKPOINT_PATH),
    "weight_heatmaps",
)


# ============================================================
# HELPERS
# ============================================================

def plot_linear_layer(
    state_dict,
    weight_name,
    bias_name,
    title,
    output_path,
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

    print(
        f"{title}: "
        f"weight shape = {weight.shape}, "
        f"bias shape = {bias.shape}"
    )

    # Use symmetric range around zero.
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
# ACTOR
# ============================================================

actor = checkpoint["actor"]

print("\nActor state dict keys:")
for key in actor.keys():
    print("   ", key)


# ------------------------------------------------------------
# Actor torso layer 1
#
# Linear(obs_dim -> 256)
# ------------------------------------------------------------

plot_linear_layer(
    actor,
    "torso._network.0.weight",
    "torso._network.0.bias",
    "Actor Torso Layer 1",
    os.path.join(
        OUTPUT_DIR,
        "actor_torso1.png",
    ),
)


# ------------------------------------------------------------
# Actor torso layer 2
#
# Linear(256 -> 256)
# ------------------------------------------------------------

plot_linear_layer(
    actor,
    "torso._network.3.weight",
    "torso._network.3.bias",
    "Actor Torso Layer 2",
    os.path.join(
        OUTPUT_DIR,
        "actor_torso2.png",
    ),
)


# ------------------------------------------------------------
# Actor torso layer 3
#
# Linear(256 -> 256)
# ------------------------------------------------------------

plot_linear_layer(
    actor,
    "torso._network.5.weight",
    "torso._network.5.bias",
    "Actor Torso Layer 3",
    os.path.join(
        OUTPUT_DIR,
        "actor_torso3.png",
    ),
)


# ------------------------------------------------------------
# Actor mean head
#
# Linear(256 -> action_dim)
# ------------------------------------------------------------

plot_linear_layer(
    actor,
    "policy_head._mean_layer.weight",
    "policy_head._mean_layer.bias",
    "Actor Mean Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_mean.png",
    ),
)


# ------------------------------------------------------------
# Actor scale / stddev head
#
# Linear(256 -> action_dim)
# ------------------------------------------------------------

plot_linear_layer(
    actor,
    "policy_head._scale_layer.weight",
    "policy_head._scale_layer.bias",
    "Actor Scale Head",
    os.path.join(
        OUTPUT_DIR,
        "actor_scale.png",
    ),
)


# ============================================================
# CRITICS
# ============================================================

critics = checkpoint["critics"]

print(
    f"\nNumber of critics: {len(critics)}"
)


for critic_index, critic in enumerate(critics):

    print(
        f"\nCritic {critic_index} state dict keys:"
    )

    for key in critic.keys():
        print("   ", key)


    # --------------------------------------------------------
    # Critic torso layer 1
    #
    # Default:
    # Linear(obs_dim + act_dim -> 512)
    # --------------------------------------------------------

    plot_linear_layer(
        critic,
        "torso._network.0.weight",
        "torso._network.0.bias",
        f"Critic {critic_index} Torso Layer 1",
        os.path.join(
            OUTPUT_DIR,
            f"critic{critic_index}_torso1.png",
        ),
    )


    # --------------------------------------------------------
    # Critic torso layer 2
    #
    # Default:
    # Linear(512 -> 512)
    # --------------------------------------------------------

    plot_linear_layer(
        critic,
        "torso._network.3.weight",
        "torso._network.3.bias",
        f"Critic {critic_index} Torso Layer 2",
        os.path.join(
            OUTPUT_DIR,
            f"critic{critic_index}_torso2.png",
        ),
    )


    # --------------------------------------------------------
    # Critic torso layer 3
    #
    # Default:
    # Linear(512 -> 256)
    # --------------------------------------------------------

    plot_linear_layer(
        critic,
        "torso._network.5.weight",
        "torso._network.5.bias",
        f"Critic {critic_index} Torso Layer 3",
        os.path.join(
            OUTPUT_DIR,
            f"critic{critic_index}_torso3.png",
        ),
    )


    # ========================================================
    # CRITIC OUTPUT HEAD
    #
    # Supports BOTH critic types in your code:
    #
    # Scalar:
    #     value_head.weight
    #
    # Distributional:
    #     distributional_head._distributional_layer.weight
    # ========================================================

    if "value_head.weight" in critic:

        plot_linear_layer(
            critic,
            "value_head.weight",
            "value_head.bias",
            f"Critic {critic_index} Scalar Output",
            os.path.join(
                OUTPUT_DIR,
                f"critic{critic_index}_output.png",
            ),
        )

    elif (
        "distributional_head._distributional_layer.weight"
        in critic
    ):

        plot_linear_layer(
            critic,
            "distributional_head._distributional_layer.weight",
            "distributional_head._distributional_layer.bias",
            f"Critic {critic_index} Distributional Output",
            os.path.join(
                OUTPUT_DIR,
                f"critic{critic_index}_output.png",
            ),
        )

    else:

        print(
            f"WARNING: Could not find output head "
            f"for critic {critic_index}"
        )


print(
    f"\nSaved heatmaps to:\n{OUTPUT_DIR}"
)
