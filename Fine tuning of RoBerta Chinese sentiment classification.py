import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline

seed = 5
torch.manual_seed(seed)  # Set PyTorch random seed
torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
np.random.seed(seed)  # Set NumPy random seed
random.seed(seed)  # Set Python built-in random seed


def load_sentence_polarity(data_path, train_ratio=0.8):
    # This task currently uses only train/test split, without a dev set.
    # train_ratio controls the split; 0.8 means 80% train and 20% test.
    all_data = []
    # categories tracks unique class labels with a set.
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['publish_time', 'content', 'date','title'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, :]:
            # polar is the sentiment class:
            #   ——2：positive
            #   ——1：neutral
            #   ——0：negative
            # sent is the corresponding sentence.
            number+=1
            if number%2==0:
                sent = value
            else:
                if value == "positive":
                    polar = 2
                elif value == "negative":
                    polar = 0
                elif value == "neutral":
                    polar = 1
            categories.add(polar)
        all_data.append((polar, sent))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data, categories



class RoBERTaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # Custom behavior can be defined here; DataLoader fetches data via __getitem__(self, index).
        # Here data is returned as self.dataset[index]; DataLoader handles index generation automatically.
        return self.dataset[index]
def sliding_window(sentence, window_size=512, stride=400):

    # List of split sub-sentences
    sub_sentences = []
    # Sentence length
    length = len(sentence)
    # Start sliding window
    start = 0
    while start < length:
        # Compute end position of current window
        end = min(start + window_size, length)
        # Split sub-sentence and append to list
        sub_sentences.append(sentence[start:end])
        # Slide window
        start += stride
    return sub_sentences


def coffate_fn(examples):
    inputs, targets = [], []
    for polar, sent in examples:

        input_text = sent
        if len(sent) > 512:
            input_text = sent[:128] + sent[-382:] # If sentence length exceeds 512, keep first 128 and last 382 tokens.
            # input_text = sent[:510]  # If sentence length does not exceed 512, use it directly
        inputs.append(input_text)
        targets.append(int(polar))

        # Use tokenizer for padding
    tokenized_inputs = tokenizer(inputs,
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt",
                                 max_length=512)

    # Ensure tensors are on the same device
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    targets = torch.tensor(targets, device=device)

    return tokenized_inputs, targets

# Training setup: define hyperparameters and global variables.

#Compute validation-set loss
def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = {key: value.to(device) for key, value in batch[0].items() if key != "labels"}
            targets = batch[1].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, targets)

            total_loss += loss.item()
        return total_loss / len(dataloader)


batch_size = 3
num_epoch = 2  # Number of training epochs
check_step = 1  # Used for mid-training checks: test and save every check_step epochs.
data_path = "./DataSet/China_Petroleum_130_records.csv"  # Dataset path
train_ratio = 0.7  # Training set ratio
learning_rate = 1e-5  # Optimizer learning rate

# Load training/testing data and number of classes
train_data, test_data, categories = load_sentence_polarity(data_path=data_path, train_ratio=train_ratio)
# Wrap train/test lists into Dataset objects for DataLoader.
train_dataset = RoBERTaDataset(train_data)
test_dataset = RoBERTaDataset(test_data)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)
print(len(test_dataloader),len(test_dataset))
#Standard pattern: cuda represents GPU.
# Use torch.cuda.is_available() to check GPU availability.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pretrained RoBERTa model and tokenizer
href="./RoBERTa"
model = AutoModelForSequenceClassification.from_pretrained(href)
tokenizer = AutoTokenizer.from_pretrained(href)



model.to(device)


optimizer = Adam(model.parameters(), learning_rate)  #Use Adam optimizer
CE_loss = nn.CrossEntropyLoss()  # Use cross-entropy loss for three-class classification

# Record current time for logging and saving
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

# Start training (model.train() standard usage).
model.train()
train_loss=[]
validate_loss=[]
for epoch in range(1, num_epoch + 1):
    # Track total loss for current epoch
    total_loss = 0
    # Use tqdm to show training progress bar in the console.
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):

        inputs, targets = batch
        inputs = {key: value.to(device) for key, value in inputs.items()}
        targets = targets.to(device)
        # Clear existing gradients
        optimizer.zero_grad()

        # Forward pass; model(inputs) is equivalent to model.forward(inputs).
        bert_output = model(**inputs)

        logits = bert_output.logits  # Extract logits
        loss = nn.CrossEntropyLoss()(logits, targets)

        # Backpropagate gradients
        loss.backward()

        # Update model parameters using backpropagated gradients
        optimizer.step()

        # Accumulate total loss; .item() extracts scalar value from tensor.
        total_loss += loss.item()
    train_loss.append(total_loss / len(train_dataloader))
    # Compute loss on validation set
    val_loss = compute_loss(model, test_dataloader, CE_loss, device)
    validate_loss.append(val_loss)

#Plot loss curves
g1=plt.figure()
plt.plot(train_loss,color='r', label='train_loss')
plt.plot(validate_loss,color='g', label='validate_loss')
plt.title('Train/validate Loss')
plt.xlabel('Epoch#')
plt.ylabel('Loss')
plt.yticks([0,0.2,0.4,0.6,0.8,1])
plt.grid()
plt.legend() #Add legend
plt.savefig("./DataSet/FineTuned_RoBERTa_LOSS")
plt.show()


#Testing phase
# acc counts correct predictions on test data.
acc = 0
tg=0
true=[]
pred=[]


def load_test(data_path):

    all_data = []
    # categories tracks unique class labels with a set.
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['publish_time', 'content', 'date','title'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, :]:
            # polar is the sentiment class:
            #   ——2：positive
            #   ——1：neutral
            #   ——0：negative
            # sent is the corresponding sentence.
            number+=1
            if number%2==0:
                sent = value
            else:
                if value == "positive":
                    polar = 2
                elif value == "negative":
                    polar = 0
                elif value == "neutral":
                    polar = 1
            categories.add(polar)
        all_data.append((polar, sent))
    test_data = all_data
    return test_data
data_path = "./DataSet/news_full_text_labeled.csv"  # Dataset path
test_data = load_test(data_path=data_path)

test_dataset = RoBERTaDataset(test_data)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)

for batch in tqdm(test_dataloader, desc=f"Testing"):
    inputs, targets = batch
    inputs = {key: value.to(device) for key, value in inputs.items()}
    targets = targets.to(device)

    with torch.no_grad():
        bert_output = model(**inputs)

        # Get logits
        logits = bert_output.logits

        # Use argmax to get predictions
        preds = logits.argmax(dim=1)
        true.append(targets.item())
        pred.append(preds.item())
        acc += (preds== targets).sum().item()
        tg+=len(targets)
# Calculate accuracy
accuracy=acc/tg
# Calculate F1 score
f1_micro = f1_score(true, pred, average='weighted')
# Calculate recall
recall = recall_score(true, pred, average='weighted')  # You can choose other average settings.
# Calculate precision
precision = precision_score(true, pred, average='weighted')  # You can choose other average settings.
# print(f"Recall: {recall:.4f}\nPrecision: {precision:.4f}")
# print(f"Accuracy: {accuracy:.4f}\nF1 score: {f1_micro:.4f}")

#Confusion matrix
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(true, pred, labels=label)
# Compute the number of true samples in each row
row_sums = cm.sum(axis=1, keepdims=True)
# Divide each confusion-matrix cell by its row total to obtain probabilities.
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/FineTuned_RoBERTa")
plt.show()




precisions = []  # Precision for the 3 classes
recalls = []  # Recall for the 3 classes
weights = [0.5, 0.3, 0.2]  # Negative class has highest weight, positive second, neutral lowest.

for i in range(len(label)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precisions.append(precision)
    recalls.append(recall)

def weighted_f1_score(precisions, recalls, weights):
    weighted_precision = sum(p * w for p, w in zip(precisions, weights))
    weighted_recall = sum(r * w for r, w in zip(recalls, weights))
    f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    return f1_score

weighted_f1 = weighted_f1_score(precisions, recalls, weights)
print(f"Recall: {recall:.4f}\nPrecision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}\nF1 score：{f1_micro:.4f}")
print(f"WeightedF1 score:{weighted_f1:.4f}\n")

