import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "./DataSet/news_full_text_labeled.csv"
df = pd.read_csv(data_path, encoding="UTF-8")
data = df.drop(labels=['publish_time',"title", 'content', 'date'], axis=1)
print(data.head())


# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# Initialize lists to store feature vectors and labels for each text.
all_features = []
all_labels = []

# Process each long text
# for text, label in zip(texts, labels):
for i in range(0,len(data)):
    text=data["news_full_text"][i]
    temp_label=data["label"][i]
    if temp_label=="positive":
        label=2
    elif temp_label=="negative":
        label=0
    else:
        label=1
    # For sentences over 512 tokens, keep first 128 and last 382 tokens.
    if len(tokenizer.tokenize(text)) > 512:
        input_text = text[:128] + text[-382:]
    else:
        input_text = text

    # Convert processed sentence into model input format
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Extract BERT text features (last hidden states).
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

    # Use CLS token as whole-sentence representation (assumes first token is CLS).
    cls_token = last_hidden_states[:, 0, :].numpy()

    # Append to feature and label lists
    all_features.append(cls_token)
    all_labels.append(label)

# Convert to NumPy arrays
all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=5)

# Initialize and train multinomial logistic regression classifier
classifier = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=1000)
classifier.fit(X_train, y_train)

# Predict and evaluate classifier performance
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


#Confusion matrix
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=label)
# Compute the number of true samples in each row
row_sums = cm.sum(axis=1, keepdims=True)
# Divide each confusion-matrix cell by its row total to obtain probabilities.
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('y_test labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/BERT_not_fine_tuned")
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
print(f"Accuracy: {accuracy:.4f}\nF1 score：{f1:.4f}")
print(f"WeightedF1 score:{weighted_f1:.4f}\n")

