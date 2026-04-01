import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")
seed = 5
torch.manual_seed(seed)  # Set PyTorch random seed
torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
np.random.seed(seed)  # Set NumPy random seed
random.seed(seed)  # Set Python built-in random seed
# Set transformers logging level to reduce unnecessary warnings; this does not affect training.
logging.set_verbosity_error()


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Define a custom model by inheriting nn.Module.
class BertSST2Model(nn.Module):

    # Initialize class
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args:
            class_size  :Specify the final number of classes for the classifier to determine the mapping dimension of the linear classifier
            pretrained_name :Used to specify the pretrained BERT model
        """
        # Superclass initialization (standard pattern).
        super(BertSST2Model, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)

        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):

        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        # Feed these three inputs into the model.
        output = self.bert(input_ids, input_tyi, input_attn_mask)


        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


def save_pretrained(model, path):
    # Save model: create directory with os, then write model file with torch.save().
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_polarity(data_path, train_ratio=0.7):

    all_data = []
    # categories tracks unique class labels with a set.
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['publish_time', 'content', 'date'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, 1:]:
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




class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):

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
        # If sentence length exceeds 512, apply sliding-window preprocessing.
        if len(sent) > 512:
            # # Split sentence with a sliding window
            # sub_sentences = sliding_window(sent)
            # inputs.extend(sub_sentences)
            # targets.extend([int(polar)] * len(sub_sentences))

            # If sentence length exceeds 512, keep first 128 and last 382 tokens.
            input_text = sent[:128] + sent[-382:]
            inputs.append(input_text)
            targets.append(int(polar))
        else:

            inputs.append(sent)
            targets.append(int(polar))

    input_dict = tokenizer(inputs,
                           padding=True,
                           truncation=True,
                           return_tensors="pt",
                           max_length=512)
    targets = torch.tensor(targets)

    return input_dict, targets


# Training setup: define hyperparameters and global variables.

#Compute validation-set loss
def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs, targets = [x.to(device) for x in batch]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


batch_size = 4
num_epoch = 2  # Number of training epochs
check_step = 1  # Used for mid-training checks: test and save every check_step epochs.
data_path = "./DataSet/China_Petroleum_130_records.csv"  # Dataset path
train_ratio = 0.7  # Training set ratio
learning_rate = 1e-5  # Optimizer learning rate

# Load training/testing data and number of classes
train_data, test_data, categories = load_sentence_polarity(data_path=data_path, train_ratio=train_ratio)
# Wrap train/test lists into Dataset objects for DataLoader.
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)

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

# Load pretrained model for Chinese data: bert-base-chinese.

pretrained_model_name = 'bert-base-chinese'
# Create BertSST2Model
model = BertSST2Model(len(categories), pretrained_model_name)

model.to(device)
# Load tokenizer matching the pretrained model
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)


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

        inputs, targets = [x.to(device) for x in batch]

        # Clear existing gradients
        optimizer.zero_grad()

        # Forward pass; model(inputs) is equivalent to model.forward(inputs).
        bert_output = model(inputs)

        # Compute loss (cross entropy).
        loss = CE_loss(bert_output, targets)

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
plt.savefig("./DataSet/BERT_LOSS")
plt.show()


#Testing phase
# acc counts correct predictions on test data.
acc = 0
tg=0
true=[]
pred=[]


def load_test(data_path):
    # This task currently uses only train/test split, without a dev set.
    # train_ratio controls the split; 0.8 means 80% train and 20% test.
    all_data = []
    # categories tracks unique class labels with a set.
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['publish_time', 'content', 'date'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, 1:]:
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

test_dataset = BertDataset(test_data)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)

for batch in tqdm(test_dataloader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    # with torch.no_grad() is standard usage here.
    # Tensor operations in this block do not track gradients, saving time and memory.
    with torch.no_grad():
        bert_output = model(inputs)

        preds=bert_output.argmax(dim=1)
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
plt.savefig("./DataSet/BERT_fine_tuned")
plt.show()


precisions = []  # Precision for the 3 classes
recalls = []  # Recall for the 3 classes
weights = [0.5, 0.3, 0.2]  # Negative class has highest weight, positive second, neutral lowest.

for i in range(len(label)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    temp_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    temp_recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precisions.append(temp_precision)
    recalls.append(temp_recall)


# Print results
print("Precisions:", precisions)
print("Recalls:", recalls)

def weighted_f1_score(precisions, recalls, weights):
    weighted_precision = sum(p * w for p, w in zip(precisions, weights))
    weighted_recall = sum(r * w for r, w in zip(recalls, weights))
    f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    return f1_score

weighted_f1 = weighted_f1_score(precisions, recalls, weights)
print(f"Recall: {recall:.4f}\nPrecision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}\nF1 score：{f1_micro:.4f}")
print(f"WeightedF1 score:{weighted_f1:.4f}\n")

