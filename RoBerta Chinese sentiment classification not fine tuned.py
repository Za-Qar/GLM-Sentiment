import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
href="./RoBERTa"
model = AutoModelForSequenceClassification.from_pretrained(href)
tokenizer = AutoTokenizer.from_pretrained(href)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

data=pd.read_csv("./DataSet/news_full_text_labeled.csv",encoding="UTF-8")
data_length=len(data)
predicted_labels=[]
test_y=data["label"]
for i in range(0,data_length):
    text=data["news_full_text"][i]
    text_length=len(text)
    if text_length>=512:
        text=text[:128]+text[-382:]
    label = classifier(text)[0]['label']
    print("No."+str(i+1)+" label："+label)
    predicted_labels.append(label)
true_labels=[]
for i in range(data_length):
    pred_label=predicted_labels[i]
    true_label=test_y[i]
    if pred_label== "negative":
        predicted_labels[i]=0
    elif pred_label == "neutral":
        predicted_labels[i]=1
    elif pred_label == "positive":
        predicted_labels[i]=2

    if true_label == "negative":
        true_labels.append(0)
    elif true_label == "neutral":
        true_labels.append(1)
    elif true_label == "positive":
        true_labels.append(2)
print(type(predicted_labels),type(true_labels))

# Calculate accuracy
accuracy=sum(1 for x, y in zip(true_labels, predicted_labels) if x == y)/data_length
# Calculate F1 score
f1_micro = f1_score(true_labels, predicted_labels, average='weighted')
# Calculate recall
recall = recall_score(true_labels, predicted_labels, average='weighted')  # You can choose other average settings.
# Calculate precision
precision = precision_score(true_labels, predicted_labels, average='weighted')  # You can choose other average settings.
print(f"Recall: {recall:.4f}\nPrecision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}\nF1 score：{f1_micro:.4f}")

#Confusion matrix
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(true_labels, predicted_labels, labels=label)
# Compute the number of true samples in each row
row_sums = cm.sum(axis=1, keepdims=True)
# Divide each confusion-matrix cell by its row total to obtain probabilities.
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/NotFineTuned_RoBERTa")
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
print(f"WeightedF1 score:{weighted_f1:.4f}\n")
