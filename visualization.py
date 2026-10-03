import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")


def plot_training_history(csv_path="Bert_Full_Weight_Fine_Tune/bert_large/history.csv", save_path="training_history.png",
                          title="Training vs Validation Loss"):
    history = pd.read_csv(csv_path)

    # long-form so seaborn can plot train and val loss as separate, labeled lines
    long_history = history.melt(
        id_vars="epoch",
        value_vars=["loss", "val_loss"],
        var_name="split",
        value_name="loss_value",
    )
    long_history["split"] = long_history["split"].map({"loss": "Train", "val_loss": "Validation"})

    epoch = history["epoch"].values
    train_loss = history["loss"].values
    val_loss = history["val_loss"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    # y-range computed up front so the gradient bands can span the full plot height
    y_min = min(train_loss.min(), val_loss.min())
    y_max = max(train_loss.max(), val_loss.max())
    padding = (y_max - y_min) * 0.1 or 0.05
    y_min, y_max = y_min - padding, y_max + padding

    # shade likely overfit/underfit regions as smooth, continuous gradients (imshow,
    # not many small rectangles — avoids visible seams between segments), drawn
    # first so the loss lines sit on top. Normalized to each zone's own span so
    # it's light at the end nearest the middle/sweet-spot, dark at the outer edge.
    train_diff = np.diff(train_loss)
    val_diff = np.diff(val_loss)
    # overfit signature: train still improving while val gets worse
    overfit_mask = (train_diff < 0) & (val_diff > 0)
    # underfit signature: both still improving together — hasn't diverged yet
    underfit_mask = (train_diff < 0) & (val_diff < 0)

    overfit_idxs = np.where(overfit_mask)[0]
    if len(overfit_idxs):
        x_start, x_end = epoch[overfit_idxs.min()], epoch[overfit_idxs.max() + 1]
        gradient = np.linspace(0, 1, 256).reshape(1, -1)  # light (middle side) -> dark (far-right edge)
        ax.imshow(gradient, extent=[x_start, x_end, y_min, y_max], aspect="auto",
                  cmap="Reds", alpha=0.3, zorder=0)

    underfit_idxs = np.where(underfit_mask)[0]
    if len(underfit_idxs):
        x_start, x_end = epoch[underfit_idxs.min()], epoch[underfit_idxs.max() + 1]
        gradient = np.linspace(1, 0, 256).reshape(1, -1)  # dark (far-left edge) -> light (middle side)
        ax.imshow(gradient, extent=[x_start, x_end, y_min, y_max], aspect="auto",
                  cmap="Blues", alpha=0.3, zorder=0)

    sns.lineplot(
        data=long_history,
        x="epoch",
        y="loss_value",
        hue="split",
        marker="o",
        ax=ax,
    )
    ax.grid(False)  # drop the default whitegrid gridlines — they clutter the shaded zones

    # sweet spot = where train and val loss cross; if they never cross, use closest approach
    diff = train_loss - val_loss
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    cross_idx = sign_changes[0] if len(sign_changes) else np.argmin(np.abs(diff))
    sweet_x, sweet_y = epoch[cross_idx], train_loss[cross_idx]
    ax.scatter(
        [sweet_x], [sweet_y],
        marker="*", s=500, color="gold", edgecolor="black", linewidth=1,
        zorder=5, label="Sweet spot",
    )

    # Zoom in on the actual range of the data instead of letting the axis
    # start at 0 — losses often cluster in a small band, and stretching the
    # axis down to 0 flattens that band into a near-straight line.
    ax.set_ylim(y_min, y_max)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # add color-meaning entries (red/blue/white) underneath the existing line/star legend
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Patch(facecolor="red", alpha=0.3, label="Overfit"),
        Patch(facecolor="blue", alpha=0.3, label="Underfit"),
        Patch(facecolor="white", edgecolor="black", label="Good fit"),
    ]
    ax.legend(handles=handles, title="", fontsize="small", markerscale=0.7, handlelength=1.2, handleheight=1.0, ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # free the figure — we make several in one run


def plot_confusion_matrix(csv_path="Bert_Full_Weight_Fine_Tune/bert_large/test_predictions.csv", save_path="confusion_matrix.png", title="Confusion Matrix (Test Set)"):
    predictions = pd.read_csv(csv_path)
    labels = ["LEFT", "CENTER", "RIGHT"]

    cm = confusion_matrix(predictions["true_label"], predictions["pred_label"], labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=True, ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# the 4 trained models: display name -> the folder its training script saves into
RUNS = {
    "BERT-large full fine-tune": "Bert_Full_Weight_Fine_Tune/bert_large",
    "BERT-small full fine-tune": "Bert_Full_Weight_Fine_Tune/bert_small",
    "LoRA top 2 layers": "Lora_Models/head_only_lora_top2",
    "LoRA all 24 layers": "Lora_Models/head_only_lora_all_layers",
}


def plot_all():
    # each run's plots are saved inside its own folder, next to its weights
    for name, folder in RUNS.items():
        history_csv = os.path.join(folder, "history.csv")
        predictions_csv = os.path.join(folder, "test_predictions.csv")
        if os.path.exists(history_csv):
            plot_training_history(history_csv, os.path.join(folder, "training_history.png"), title=f"{name}: Training vs Validation Loss")
            print(f"[{name}] saved {folder}/training_history.png")
        else:
            print(f"[{name}] skipped training plot — {history_csv} not found (run its training script first)")
        if os.path.exists(predictions_csv):
            plot_confusion_matrix(predictions_csv, os.path.join(folder, "confusion_matrix.png"),
                                  title=f"{name}: Confusion Matrix (Test Set)")
            print(f"[{name}] saved {folder}/confusion_matrix.png")
        else:
            print(f"[{name}] skipped confusion matrix — {predictions_csv} not found (run its training script first)")


if __name__ == "__main__":
    plot_all()
