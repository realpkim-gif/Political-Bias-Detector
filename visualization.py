import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")


def plot_training_history(csv_path="model_patience_5.csv", save_path="training_history.png"):
    history = pd.read_csv(csv_path)

    # long-form so seaborn can plot train and val loss as separate, labeled lines
    long_history = history.melt(
        id_vars="epoch",
        value_vars=["loss", "val_loss"],
        var_name="split",
        value_name="loss_value",
    )
    long_history["split"] = long_history["split"].map({"loss": "Train", "val_loss": "Validation"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=long_history,
        x="epoch",
        y="loss_value",
        hue="split",
        marker="o",
        ax=ax,
    )

    # Zoom in on the actual range of the data instead of letting the axis
    # start at 0 — losses often cluster in a small band, and stretching the
    # axis down to 0 flattens that band into a near-straight line.
    y_min = long_history["loss_value"].min()
    y_max = long_history["loss_value"].max()
    padding = (y_max - y_min) * 0.1 or 0.05
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(title="")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)



def plot_bias_variance(csv_path="model_patience_5.csv", save_path="bias_variance_real.png"):
    # Uses actual training history: train_loss as a stand-in for "bias" (how well
    # the model fits what it's seeing) and val_loss as a stand-in for "variance"
    # (how well that generalizes) — not a rigorous bias/variance decomposition,
    # just the same red/blue/sweet-spot framing applied to your real numbers.
    history = pd.read_csv(csv_path)

    epoch = history["epoch"].values
    bias = history["loss"].values
    variance = history["val_loss"].values

    # sweet spot = where the two curves cross; if they never cross, use closest approach
    diff = bias - variance
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    cross_idx = sign_changes[0] if len(sign_changes) else np.argmin(np.abs(diff))
    sweet_x, sweet_y = epoch[cross_idx], bias[cross_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, bias, color="red", linewidth=2, marker="o", label="Bias (train loss)")
    ax.plot(epoch, variance, color="blue", linewidth=2, marker="o", label="Variance (val loss)")
    ax.scatter(
        [sweet_x], [sweet_y],
        marker="*", s=500, color="gold", edgecolor="black", linewidth=1,
        zorder=5, label="Sweet spot",
    )

    ax.set_title("Bias-Variance Tradeoff (Real Training Data)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def plot_confusion_matrix(csv_path="test_predictions.csv", save_path="confusion_matrix.png"):
    predictions = pd.read_csv(csv_path)
    labels = ["LEFT", "CENTER", "RIGHT"]

    cm = confusion_matrix(predictions["true_label"], predictions["pred_label"], labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=True, ax=ax,
    )

    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    plot_training_history()
    plot_bias_variance()
    plot_confusion_matrix()
