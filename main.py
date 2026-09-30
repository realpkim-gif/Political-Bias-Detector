from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.functional import accuracy, precision, recall, f1_score
import re

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
bert_model = AutoModel.from_pretrained("google-bert/bert-large-uncased").to(device)
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

num_epochs=50

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
    # no blanket no_grad here: the last few BERT layers are unfrozen and need
    # gradients during training. Callers wrap this in torch.no_grad() for eval.

    return outputs.pooler_output  # (batch, 1024) — BERT's own [CLS]-based summary of the sequence
    #pooler_output = tanh(Linear(last_hidden_state[:, 0, :]))
"""
This is exactly the setup BERT's original paper and GLUE benchmark used, 
and it's what HuggingFace's own BertForSequenceClassification does by default for classification fine-tuning.
"""

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #initiallize
        super(SimpleNeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully Connected Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Fully Connected Layer 2
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Outputs logits; use CrossEntropyLoss (applies softmax)

    def forward(self, test_input): #internally called
        x = get_embedding(test_input, device)  # BERT embedding is the starting input
        x = F.relu(self.fc1(x)) #maintains shape
        x = F.relu(self.fc2(x)) #maintains shape
        scores = self.fc3(x)  # (batch, 3) raw scores
        return scores


# BERT-large embeddings are 1024-dim; 3 classes: LEFT, CENTER, RIGHT (from get_embeddings)
final_model = SimpleNeuralNet(input_size=1024, hidden_size=256, num_classes=3).to(device)

# inverse-frequency class weights so the loss doesn't ignore CENTER (the minority class)
class_counts = y_train.value_counts().sort_index()
class_weights = torch.tensor(
    (len(y_train) / (3 * class_counts)).values, dtype=torch.float32, device=device
)
loss_function = nn.CrossEntropyLoss(weight=class_weights)

trainable_bert_params = [p for p in bert_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(list(final_model.parameters()) + trainable_bert_params)

loader = DataLoader(
    dataset=TextLabelDataset(X_train, y_train),
    batch_size=4,
    shuffle=True
)

def predict_in_batches(model, texts, labels):
    train_loader = DataLoader(TextLabelDataset(texts, labels), batch_size=4, shuffle=False)
    all_scores = []
    all_labels = []
    # same fp16 speed/memory benefit as training; no GradScaler needed here
    # since there's no backward() pass during evaluation
    with torch.amp.autocast("cuda"):
        for batch_texts, batch_labels in train_loader:
            all_scores.append(model(batch_texts))
            all_labels.append(batch_labels)
    return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0)


#returns batch_idx, (test_input, test_output)

#applying optimization
def train(model, optimizer, loss_function, train_loader, patience, train_name):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"epoch": [], "loss": [], "val_loss": []}

    # AMP = autocast (forward pass, picks fp16/fp32 per-op, used in both train and eval) + GradScaler (backward pass only, pure multiply/divide, no dtype casting)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(num_epochs):
        model.train()
        bert_model.train()  # unfrozen BERT layers need dropout active during training too
        running_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (test_input, test_output) in enumerate(train_loader):
            optimizer.zero_grad()

            # autocast: runs the ops inside this block in fp16 where safe (matmuls,
            # the big BERT computations), while keeping precision-sensitive ops
            # (like the loss) in fp32 automatically. Weights themselves stay fp32 —
            # only certain safe operations get a temporary fp16 copy of their inputs.
            # Dangerous opperations like sigmoid keep/upscale to fp32 and output fp32
            with torch.amp.autocast("cuda"):
                output = model(test_input)
                loss = loss_function(output, test_output.to(device))

            running_loss += loss.item()
            scaler.scale(loss).backward()  # multiply loss up (no dtype change) before backward() so small gradients don't underflow to zero
            scaler.step(optimizer)  # divide gradients back down (no dtype change), then optimizer.step() with correct-sized update
            scaler.update()  # lower on detected overflow (real evidence); raise after stable streak (blind guess, underflow is never directly detectable)

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch + 1} batch {batch_idx}/{num_batches} loss={loss.item():.4f}")

        avg_train_loss = running_loss / num_batches

        #evaluate with val
        model.eval()
        bert_model.eval()
        with torch.no_grad():
            val_scores, val_labels = predict_in_batches(model, list(X_val), list(y_val))
            val_loss = loss_function(val_scores, val_labels.to(device)).item()

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        print(f"--- Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} ---")

        #early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # save training history once training is finished (not every epoch)
    hist_data = pd.DataFrame(history)
    hist_data.to_csv(f"{train_name}.csv", index=False)


train(final_model, optimizer, loss_function, loader, 5, "model_patience_5")
torch.save(final_model.state_dict(), 'model_patience_5.pt')
torch.save(bert_model.state_dict(), 'bert_finetuned_patience_5.pt')  # unfrozen layers changed too


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
    test_metrics.to_csv("test_metrics.csv", index=False)

    # save per-example true/predicted labels for the confusion matrix
    predictions = pd.DataFrame({
        "true_label": y_test_true.cpu().numpy(),
        "pred_label": pred.cpu().numpy(),
    })
    predictions.to_csv("test_predictions.csv", index=False)

