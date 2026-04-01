import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
seed = 5
np.random.seed(seed)  # Set NumPy random seed
random.seed(seed)  # Set Python built-in random seed
# pred20=pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRST'))
# feature_data=pred20.columns
# pred20.loc[1,feature_data[0]]=1
# if pred20.loc[1,"A"]==1:
#     print("equals 1")
# print(pred20,len(feature_data))


filePath="./DataSet/news_full_text_labeled.csv"
df=pd.read_csv(filePath,encoding="UTF-8")
data=df.drop(labels=['title','publish_time','content','date'],axis=1)

recallss=[]
precisionss=[]
accuracys=[]
f1_micros=[]
weighted_f1s=[]
# paths=["GLM2-1.csv","GLM3-1.csv","GLM2-3.csv","GLM3-3.csv","GLM2-4.csv","GLM3-4.csv","GLM2-5.csv","GLM3-5.csv","GLM2-6.csv","GLM3-6.csv","GLM2-integration.csv","GLM3-integration.csv"]
paths=["GLM3-1-10jis.csv","GLM3-2jis .csv","GLM2-2jis.csv","GLM3-5json (1).csv","GLM2-5jis.csv"]
#
for path in paths:
    pred20=pd.read_csv('News/'+path, encoding='utf-8')
    pred20=pred20.drop(labels=['item'],axis=1)
    # Alternative: drop index and selected vote columns if needed.

    #'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T'
    length=len(data)
    pred=pd.DataFrame(pred20)
    # print(pred.iloc[:, :-1])
    df=pred.iloc[:, :-1]
    for i in range(0,length):
        counts = {"positive": 0, "negative": 0, "neutral": 0}  # Initialize counters
        for value in pred20.iloc[i, :-1]:
            if value[0:2]== "positive":
                counts["positive"] += 1
            elif value[0:2] == "negative":
                counts["negative"] += 1
            elif value[0:2] == "neutral":
                counts["neutral"] += 1
            else:
                continue
        # print(counts)
        if counts["positive"]>counts["negative"] and counts["positive"]>counts["neutral"]:
            pred20.loc[i,"label"]="positive"
        elif counts["negative"]>counts["positive"] and counts["negative"]>counts["neutral"]:
            pred20.loc[i,"label"]="negative"
        else :
            pred20.loc[i,"label"]="neutral"
    predicted_labels=pred20["label"]
    test_y=data["label"]

    # Calculate accuracy
    accuracy=np.sum(data["label"]==pred20["label"])/length

    # Calculate F1 score
    f1_micro = f1_score(test_y, predicted_labels, average='weighted')

    # Calculate recall
    recall = recall_score(test_y, predicted_labels, average='weighted')  # You can choose other average settings.

    # Calculate precision
    precision = precision_score(test_y, predicted_labels, average='weighted')  # You can choose other average settings.


    true=[]
    pred=[]
    for i in range(length):
        if data["label"][i]=="negative":
            true.append(0)
        elif data["label"][i]=="neutral":
            true.append(1)
        elif data["label"][i]=="positive":
            true.append(2)
        if pred20.loc[i,"label"]=="negative":
            pred.append(0)
        elif pred20.loc[i,"label"]=="positive":
            pred.append(2)
        else:
            pred.append(1)
    labels=["Bearish","Neutral","Bullish"]
    label=[0,1,2]

    # #Output misclassified labels
    # for i in range(len(true)):
    #     if true[i]!=pred[i]:
    #         print(i)

    cm = confusion_matrix(true, pred, labels=label)
    # print("cm:{}".format(cm))

    # Compute the number of true samples in each row
    row_sums = cm.sum(axis=1, keepdims=True)

    # Divide each confusion-matrix cell by its row total to obtain probabilities.
    cm_prob = cm / row_sums
    ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig("./News/"+path[:-4]+".png")
    plt.show()

    precisions = []  # Precision for the 3 classes
    recalls = []  # Recall for the 3 classes
    weights = [0.5, 0.3, 0.2]  # Negative class has highest weight, positive second, neutral lowest.

    # Define helper to compute TP, FP, and FN for each class.
    def calculate_tp_fp_fn(cm, class_label):
        class_index = labels.index(class_label)
        tp = cm[class_index, class_index]
        fp = np.sum(cm[:, class_index]) - tp
        fn = np.sum(cm[class_index, :]) - tp
        return tp, fp, fn

    def weighted_f1_score(precisions, recalls, weights):
        weighted_precision = sum(p * w for p, w in zip(precisions, weights))
        weighted_recall = sum(r * w for r, w in zip(recalls, weights))

        f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

        return f1_score

    # Compute precision for each class
    def calculate_precision(tp, fp):
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    # Compute recall for each class
    def calculate_recall(tp, fn):
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)

    # Compute TP, FP, and FN for each class
    for class_label in labels:
        tp, fp, fn = calculate_tp_fp_fn(cm, class_label)
        # print(f"Class: {class_label}")
        # print("True positives:", tp)
        # print("False positives:", fp)
        # print("False negatives:", fn)

        # Compute precision and recall for each class
        precision_neg = calculate_precision(tp, fp)
        recall_neg = calculate_recall(tp, fn)

        precisions.append(precision_neg)
        recalls.append(recall_neg)


    # Calculate F1 score
    weighted_f1 = weighted_f1_score(precisions, recalls, weights)
    print(path)
    print(f"Recall: {recall*100:.2f}%\nPrecision: {precision*100:.2f}%")
    print(f"Accuracy: {accuracy*100:.2f}%\nF1 score：{f1_micro*100:.2f}%")
    print(f"WeightedF1 score:{weighted_f1*100:.2f}%\n")
    recallss.append(round(recall*100,2))
    precisionss.append(round(precision*100,2))
    accuracys.append(round(accuracy*100,2))
    f1_micros.append(round(f1_micro*100,2))
    weighted_f1s.append(round(weighted_f1*100,2))

# for i in range(len(weighted_f1s)):
#     print(f"GLM3:{weighted_f1s[i]:.4f}\tGLM2:{weighted_f1s[1+i]:.4f}")

    # # The method below can also compute weighted F1 from the confusion matrix.
    # precisions = []  # Precision for the 3 classes
    # recalls = []  # Recall for the 3 classes
    # for i in range(len(label)):
    #     tp = cm[i, i]
    #     fp = cm[:, i].sum() - tp
    #     fn = cm[i, :].sum() - tp
    #
    #     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    #
    #     precisions.append(precision)
    #     recalls.append(recall)
    # weighted_f1 = weighted_f1_score(precisions, recalls, weights)
    # print(f"Weighted value:{weighted_f1:.4f}\n")

















# print(precisionss)
# print(recallss)
#
# print(accuracys)
#
# print(f1_micros)
#
# print(weighted_f1s)

# # Set bar width
# bar_width = 0.5
# bar_width_total = 2.5  # Total bar-chart width
#
# # Set bar positions
# models=["GLM3-1","GLM2-1","GLM3-3","GLM2-3","GLM3-4","GLM2-4","GLM3-5","GLM2-5","GLM3-6","GLM2-6","GLM3-integration","GLM2-integration"]
# rr=[3,6,9,12,15,18,21,24,27,30,33,36]
# r1 = [x - bar_width*2 for x in rr]
# r2 = [x - bar_width for x in rr]
# r3 = [x  for x in rr]
# r4 = [x + bar_width for x in rr]
# r5 = [x + bar_width*2 for x in rr]
#
# # Convert RGB to Hex
# def rgb_to_hex(r, g, b):
#     return '#{:02x}{:02x}{:02x}'.format(r, g, b)
#
# precision_color = rgb_to_hex(140, 205, 191)
# recall_color = rgb_to_hex(205, 224, 165)
# accuracy_color = rgb_to_hex(249, 219, 149)
# f1_micro_color = rgb_to_hex(239, 123, 118)
# weighted_f1_color = rgb_to_hex(197, 168, 206)
#
# plt.yticks(np.arange(35, 80, 10))
#
#
# # Create bar chart
# plt.figure(figsize=(14, 8))
# plt.bar(r1, recallss, color=recall_color, width=bar_width, edgecolor='grey', label='Recall')
# plt.bar(r2, precisionss, color=precision_color, width=bar_width, edgecolor='grey', label='Precision')
# plt.bar(r3, accuracys, color=accuracy_color, width=bar_width, edgecolor='grey', label='Accuracy')
# plt.bar(r4, f1_micros, color=f1_micro_color, width=bar_width, edgecolor='grey', label='F1_score')
# plt.bar(r5, weighted_f1s, color=weighted_f1_color, width=bar_width, edgecolor='grey', label='Weighted_F1')

# # Set y-axis range
# plt.ylim(35, 80)  # Set y-axis start to 0.5 and end to 1.0
#
# # Add labels
# plt.xlabel('Models', fontweight='bold')
# plt.ylabel('Scores', fontweight='bold')
# plt.xticks(rr, models)
# plt.title('Comparison of model Performance Metrics')

# # Show height of each bar
# for r, precision, recall, accuracy, f1_micro, weighted_f1 in zip(r1, precisionss, recallss, accuracys, f1_micros, weighted_f1s):
#     plt.text(r, recall, '{:.2f}'.format(recall), ha='center', va='bottom', rotation=90)
#     plt.text(r + bar_width, precision, '{:.2f}'.format(precision), ha='center', va='bottom', rotation=90)
#     plt.text(r + 2 * bar_width, accuracy, '{:.2f}'.format(accuracy), ha='center', va='bottom', rotation=90)
#     plt.text(r + 3 * bar_width, f1_micro, '{:.2f}'.format(f1_micro), ha='center', va='bottom', rotation=90)
#     plt.text(r + 4 * bar_width, weighted_f1, '{:.2f}'.format(weighted_f1), ha='center', va='bottom', rotation=90)

# Show legend
# plt.legend()
# plt.grid(ls=":",color="gray",alpha=0.5)
# Display chart
# plt.show()
