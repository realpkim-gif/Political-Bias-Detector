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

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = AutoModel.from_pretrained("google-bert/bert-large-uncased").to(device)
model.eval() #turn off drop out layers to make stable

for param in model.parameters(): #no backprop
    param.requires_grad = False

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
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


#End of data preprocessing


def get_embedding(text, device): #(using BERT)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs) #** unpacks input/attention mask dictionary into two lists.

    return outputs.last_hidden_state.mean(dim=1)  # (batch, 1024) tensor, stays on device

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

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(final_model.parameters())  # only the classifier head is trainable; BERT is frozen

train_loader = DataLoader(
    dataset=TextLabelDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)
#returns batch_idx, (test_input, test_output)

#applying optimization
def train(model, optimizer, loss_function, train_loader, patience, train_name):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"epoch": [], "loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (test_input, test_output) in enumerate(train_loader):
            output = model(test_input)
            loss = loss_function(output, test_output.to(device))
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / len(train_loader)

        #evaluate with val
        model.eval()
        with torch.no_grad():
            val_scores = model(X_val.tolist())
            val_loss = loss_function(val_scores, torch.tensor(y_val.tolist(), device=device)).item()

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        #early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        hist_data = pd.DataFrame(history)
        hist_data.to_csv(f"{train_name}.csv", index=False)

final_model.train()

train(final_model, optimizer, loss_function, train_loader, 5, "model_patience_5")
torch.save(final_model.state_dict(), 'model_patience_5.pt')


# fresh model + optimizer, so this run doesn't continue from the patience=5 run above
final_model = SimpleNeuralNet(input_size=1024, hidden_size=256, num_classes=3).to(device)
optimizer = torch.optim.Adam(final_model.parameters())
final_model.train()

train(final_model, optimizer, loss_function, train_loader, 10, "model_patience_10")
torch.save(final_model.state_dict(), 'model_patience_10.pt')



#To convert raw logit to id
id_to_label = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}

#End of train and testing model
final_model.eval()                    # dropout off
with torch.no_grad():          # no gradient tracking for specific (diff way than BERT but same thing, only in that block with this)
    #with is try and finally (to close) but simpler
    inputs = X_test.tolist()
    scores = final_model(inputs)        # (batch, 3) raw scores
    pred = scores.argmax(dim=1)   # 0/1/2 = LEFT/CENTER/RIGHT

    y_true = torch.tensor(y_test.tolist(), device=device)

    acc = accuracy(pred, y_true, task="multiclass", num_classes=3)
    prec = precision(pred, y_true, task="multiclass", num_classes=3, average="macro")
    rec = recall(pred, y_true, task="multiclass", num_classes=3, average="macro")
    f1 = f1_score(pred, y_true, task="multiclass", num_classes=3, average="macro")

    print(acc, prec, rec, f1)
