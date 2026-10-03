import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.functional import accuracy, precision, recall, f1_score

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

MODEL_NAME = "google/bert_uncased_L-4_H-512_A-8"  # Google's "BERT-Small": 4 layers, hidden 512, 28.8M params (was google-bert/bert-large-uncased)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
# full fine-tune: every BERT parameter is trainable (requires_grad=True by default)

def load_allsides_data():
    # This dataset's files have mixed encodings (mostly utf-8, some cp1252),
    # so the generic `datasets` text loader can't decode all of them with one setting.
    zip_path = hf_hub_download("valurank/PoliticalBias_AllSides_Txt", "AllSides.zip", repo_type="dataset")
    label_map = {"Left Data": "LEFT", "Center Data": "CENTER", "Right Data": "RIGHT"}
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            folder = name.split("/")[1]
            label = label_map.get(folder)
            if label is None:
                continue
            raw = z.read(name)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="replace")
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)

df = load_allsides_data()
df.dropna(inplace=True)

# drop extreme length outliers (likely corrupted/concatenated files, not real articles)
text_lengths = df["text"].str.len()
df = df[text_lengths < text_lengths.quantile(0.99)]
df = df[df["text"].str.contains(r"\S", regex=True, na=False)]
df.drop_duplicates(subset="text", inplace=True)

df.reset_index(drop=True, inplace=True)

label_to_id = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
df["label"] = df["label"].map(label_to_id)
print(df.shape)

max_lengths_all_columns = df.astype(str).map(len).max()
print("max length", max_lengths_all_columns)

X=df["text"]
Y=df["label"]

# Split data into 80% training, 10% validation, 10% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42
)

df.to_csv('data_finetune.csv', index=False)

num_epochs=80

#Dataloader needs the index and data/lable for test, need to make a class like this for pytorch (internally calls these methods)
class TextLabelDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


#End of data preprocessing


def get_embedding(text, device): #(using BERT)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    outputs = bert_model(**inputs) #** unpacks input/attention mask dictionary into two lists.
    # no blanket no_grad here: this is a full fine-tune, so all of BERT needs
    # gradients during training. Callers wrap this in torch.no_grad() for eval.

    return outputs.pooler_output  # (batch, 1024) — BERT's own [CLS]-based summary of the sequence
    #pooler_output = tanh(Linear(last_hidden_state[:, 0, :]))
    # This is the same setup BERT's original paper and GLUE benchmark used, and
    # what HuggingFace's BertForSequenceClassification does by default.

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #initiallize
        super(SimpleNeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully Connected Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Fully Connected Layer 2
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Outputs logits; use CrossEntropyLoss (applies softmax)
        self.dropout = nn.Dropout(p=0.3)  # regularization; only active during model.train(), off during eval

    def forward(self, test_input): #internally called
        x = get_embedding(test_input, device)  # BERT embedding is the starting input
        x = F.relu(self.fc1(x)) #maintains shape
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) #maintains shape
        x = self.dropout(x)
        scores = self.fc3(x)  # (batch, 3) raw scores
        return scores


# BERT-Small embeddings are 512-dim (large was 1024) — read from the model config instead of hardcoding;
# 3 classes: LEFT, CENTER, RIGHT (from get_embeddings)
final_model = SimpleNeuralNet(input_size=bert_model.config.hidden_size, hidden_size=256, num_classes=3).to(device)

# inverse-frequency class weights so the loss doesn't ignore CENTER (the minority class)
class_counts = y_train.value_counts().sort_index()
class_weights = torch.tensor(
    (len(y_train) / (3 * class_counts)).values, dtype=torch.float32, device=device
)
loss_function = nn.CrossEntropyLoss(weight=class_weights)

trainable_bert_params = [p for p in bert_model.parameters() if p.requires_grad]
# lr=2e-5: Adam's default (1e-3) is ~50x too large for fine-tuning a pretrained
# transformer — it wrecked BERT's pretrained weights and collapsed the model
# into always predicting the majority class within the first epoch
# AdamW vs Adam: decouples weight decay from the gradient update instead of blending them together — standard practice for larger models like BERT
base_lr = 2e-5
optimizer = torch.optim.AdamW(list(final_model.parameters()) + trainable_bert_params, lr=base_lr)

loader = DataLoader(
    dataset=TextLabelDataset(X_train, y_train),
    batch_size=4,
    shuffle=True
)

# linear warmup over a fixed 1 epoch's worth of steps (ramps 0 -> base_lr), applied
# manually per-batch in train(). This is purely a stability measure — protects
# BERT's pretrained weights from large, destructive early updates — not a lever
# for better accuracy. Once past that risk window, a longer warmup adds no benefit.
# Fixed to len(loader) rather than a % of num_epochs
# so it doesn't balloon just because num_epochs is set high for early-stopping
# headroom — num_epochs is a safety ceiling, not a real training-length estimate.
# After warmup, LR holds at base_lr until validation accuracy fails to improve —
# decay only starts once accuracy actually stalls, instead of decaying from the
# first step regardless of whether the model is still improving. Epoch-level
# accuracy (not per-batch) is used since batch_size=4 makes per-batch accuracy
# too noisy (4 examples) to react to meaningfully.
#
# Two-stage patience, tracked manually in train() since ReduceLROnPlateau can't
# change its patience mid-run: the FIRST decay uses patience=1 (react fast to catch
# accuracy turning bad early), every decay AFTER that uses patience=3 (looser, so
# it doesn't keep halving the lr every single epoch once it's already reacted once).
warmup_steps = len(loader) * 1
LR_DECAY_FACTOR = 0.5
MIN_LR = 1e-7

def predict_in_batches(model, texts, labels):
    train_loader = DataLoader(TextLabelDataset(texts, labels), batch_size=4, shuffle=False)
    all_scores = []
    all_labels = []
    # bf16: same speed/memory benefit as fp16, but no GradScaler needed at all
    # (bf16 has fp32's range, so there's no overflow/underflow to guard against).
    # autocast only casts "safe" ops (matmuls) to bf16; unsafe ops stay at their
    # original dtype or get upcast to at least fp32 — never cast down.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch_texts, batch_labels in train_loader:
            all_scores.append(model(batch_texts))
            all_labels.append(batch_labels)
    # cast back to fp32: autocast returns bf16-dtype scores, but loss_function's
    # class_weights tensor is fp32, and this is called outside any autocast block
    # (no automatic reconciliation) — without this, dtype mismatch crashes cross_entropy
    return torch.cat(all_scores, dim=0).float(), torch.cat(all_labels, dim=0)


# Both of these control lr, grouped together for readability, but run at different
# granularities: warmup is called per-batch (needs global_step, which only
# increments inside the batch loop); decay is called per-epoch (needs val_accuracy,
# which only exists once validation has run over the whole val set for that epoch).

def apply_warmup(optimizer, global_step, warmup_steps, base_lr):
    # linear warmup: ramp lr from 0 -> base_lr over the first warmup_steps
    # batches. Once warmup finishes, this stops touching lr entirely, so
    # the decay logic's accuracy-plateau reductions aren't fought/overwritten.
    if global_step <= warmup_steps:
        print(f"[WARMUP] step {global_step}/{warmup_steps}")
        # global_step/warmup_steps is the fraction of warmup completed so far,
        # not a coincidence: at global_step == warmup_steps the fraction is
        # exactly 1, so lr lands exactly on base_lr — never over or under.
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * global_step / warmup_steps
            #when global_step = warmup then base_lr is achieved (at last batch) then decay phase


def apply_accuracy_decay(optimizer, val_accuracy, best_val_accuracy, epochs_without_lr_improvement, has_decayed_once, lr_patience):
    # two-stage lr decay: patience=1 for the first decay (react fast), patience=3
    # for every decay after that (don't keep halving every single epoch)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_lr_improvement = 0
    else:
        epochs_without_lr_improvement += 1
        if epochs_without_lr_improvement >= lr_patience:
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"] * LR_DECAY_FACTOR, MIN_LR)
            epochs_without_lr_improvement = 0
            if not has_decayed_once:
                has_decayed_once = True
                lr_patience = 3  # loosen up after the first reaction
    return best_val_accuracy, epochs_without_lr_improvement, has_decayed_once, lr_patience


#applying optimization
def train(model, optimizer, loss_function, train_loader, patience, train_name):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"epoch": [], "loss": [], "val_loss": [], "val_accuracy": []}
    os.makedirs(train_name, exist_ok=True)  # creates Bert_Full_Weight_Fine_Tune/<run name>/ (and the top folder if missing)
    global_step = 0  # counts batches across the whole run, for the warmup ramp below

    # two-stage lr decay: patience=1 for the first decay (react fast), patience=3
    # for every decay after that (don't keep halving every single epoch)
    best_val_accuracy = float("-inf")
    epochs_without_lr_improvement = 0
    has_decayed_once = False
    lr_patience = 1

    # bf16 instead of fp16: same speed/memory win, but bf16 has fp32's full exponent
    # range, so there's no overflow/underflow risk and no GradScaler needed at all
    for epoch in range(num_epochs):
        model.train()
        bert_model.train()  # unfrozen BERT layers need dropout active during training too
        running_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (test_input, test_output) in enumerate(train_loader):
            optimizer.zero_grad()

            # autocast only casts to bf16 for ops on its "safe" list (matmuls, the big
            # BERT computations). Unsafe ops either stay at whatever dtype they already
            # were, or get upcast to at least fp32 (e.g. the loss) — never cast down.
            # Weights themselves stay fp32 regardless; only op inputs get a temp bf16 copy.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(test_input)
                loss = loss_function(output, test_output.to(device))

            running_loss += loss.item()
            # dtype here follows each op's original forward-pass dtype (bf16 for the autocast ops above), not this line's position outside the with block.
            # So not the default fp since forward used bf.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=1.0)  # cap extreme gradients before they hit the optimizer
            optimizer.step()

            # stop incrementing once past warmup_steps — nothing downstream reads the
            # exact value, only whether it's <= or > warmup_steps, and that comparison
            # result is already locked in permanently once it crosses the threshold once
            if global_step <= warmup_steps:
                global_step += 1
            apply_warmup(optimizer, global_step, warmup_steps, base_lr)

            print(f"  epoch {epoch + 1} batch {batch_idx}/{num_batches} loss={loss.item():.4f}")

        avg_train_loss = running_loss / num_batches

        #evaluate with val
        model.eval()
        bert_model.eval()
        with torch.no_grad():
            val_scores, val_labels = predict_in_batches(model, list(X_val), list(y_val))
            val_loss = loss_function(val_scores, val_labels.to(device)).item()
            val_accuracy = accuracy(val_scores.argmax(dim=1), val_labels.to(device), task="multiclass", num_classes=3).item()

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # only react once warmup is done — during warmup, lr is fully controlled by
        # the ramp above, so a reduction here would just get overwritten by the
        # next batch's warmup step anyway

        if global_step > warmup_steps:
            print(f"[DECAY] epoch {epoch + 1}: val_accuracy={val_accuracy:.4f} best={best_val_accuracy:.4f} patience={lr_patience}")
            best_val_accuracy, epochs_without_lr_improvement, has_decayed_once, lr_patience = apply_accuracy_decay(
                optimizer, val_accuracy, best_val_accuracy, epochs_without_lr_improvement, has_decayed_once, lr_patience
            ) #apply decay with patience of 3 (original function)

        print(f"--- Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e} ---")

        #early stop — only once lr has bottomed out at MIN_LR; before that, the lr
        # decay above still has room to try a gentler rate, so don't give up yet
        current_lr = optimizer.param_groups[0]["lr"]
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # save the best checkpoint as soon as we see it — training can keep
            # running a long time after this (early stop is gated on lr reaching
            # MIN_LR), and the model can overfit in the meantime. Without this,
            # we'd only ever have access to whatever the LAST epoch looked like.
            torch.save(model.state_dict(), os.path.join(train_name, "head.pt"))
            torch.save(bert_model.state_dict(), os.path.join(train_name, "bert.pt"))
        elif current_lr <= MIN_LR:
            epochs_without_improvement += 1 #apply decay with patience of 5 once reach min (special not related to the original function of
            # patience of 3 decay and dividing by 2.
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # save training history once training is finished (not every epoch)
    hist_data = pd.DataFrame(history)
    hist_data.to_csv(os.path.join(train_name, "history.csv"), index=False)


train_name = os.path.join("Bert_Full_Weight_Fine_Tune", "bert_small")  # all outputs go in Bert_Full_Weight_Fine_Tune/bert_small/ (own folder, can't overwrite main.py's)
train(final_model, optimizer, loss_function, loader, 5, train_name)
# checkpoints are saved inside train() as soon as a new best val_loss is seen —
# nothing to save here, that would overwrite the best with the final epoch's state

# reload the best-val_loss checkpoint before testing — without this, the test
# block below would just evaluate whatever's left in memory (the LAST epoch),
# not the best one that got saved to disk during training.
final_model.load_state_dict(torch.load(os.path.join(train_name, "head.pt"), map_location=device))
bert_model.load_state_dict(torch.load(os.path.join(train_name, "bert.pt"), map_location=device))
print(f"Reloaded best checkpoint (by val_loss) from {train_name}/head.pt and {train_name}/bert.pt for testing")


#End of train and TEST model
final_model.eval()                    # dropout off
bert_model.eval()
with torch.no_grad():          # no gradient tracking for specific (diff way than BERT but same thing, only in that block with this)
    #with is try and finally (to close) but simpler
    inputs = list(X_test)
    scores, y_test_true = predict_in_batches(final_model, inputs, list(y_test))        # (batch, 3) raw scores

    pred = scores.argmax(dim=1)   # 0/1/2 = LEFT/CENTER/RIGHT

    y_test_true = y_test_true.to(device)

    acc = accuracy(pred, y_test_true, task="multiclass", num_classes=3)
    prec = precision(pred, y_test_true, task="multiclass", num_classes=3, average="macro")
    rec = recall(pred, y_test_true, task="multiclass", num_classes=3, average="macro")
    f1 = f1_score(pred, y_test_true, task="multiclass", num_classes=3, average="macro")

    print(acc, prec, rec, f1)

    test_metrics = pd.DataFrame([{
        "accuracy": acc.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "f1": f1.item(),
    }])
    test_metrics.to_csv(os.path.join(train_name, "test_metrics.csv"), index=False)

    # save per-example true/predicted labels for the confusion matrix
    predictions = pd.DataFrame({
        "true_label": y_test_true.cpu().numpy(),
        "pred_label": pred.cpu().numpy(),
    })
    predictions.to_csv(os.path.join(train_name, "test_predictions.csv"), index=False)

