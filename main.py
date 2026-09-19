from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device="cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = AutoModel.from_pretrained("google-bert/bert-large-uncased").to(device)
model.eval()

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
print(df.shape)

df.to_csv('data_finetune.csv', index=False)

def get_embedding(text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    #** unpacks input/attention mask dictionary into two lists.
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully Connected Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Fully Connected Layer 2
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Outputs logits; use CrossEntropyLoss (applies softmax)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# BERT-large embeddings are 1024-dim; 3 classes: LEFT, CENTER, RIGHT
final_model = SimpleNeuralNet(input_size=1024, hidden_size=256, num_classes=3)

# Create a mock batch of data (batch size of 4, 10 features each)
#mock_input = torch.randn(4, 10)

# Run the forward pass
#predictions = model(mock_input)

# check the actual field names first
print(df['text'][0])