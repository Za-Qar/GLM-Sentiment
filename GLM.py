import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
import random
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
seed = 5
torch.manual_seed(seed)  # Set PyTorch random seed
torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs
np.random.seed(seed)  # Set NumPy random seed
random.seed(seed)  # Set Python built-in random seed
from pylab import mpl

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()
print("Running")
response, history = model.chat(tokenizer, "Hello", history=[])
print(response)

print("label")
filePath = "./DataSet/news_full_text_labeled.csv"
df = pd.read_csv(filePath, encoding="UTF-8")
data = df.drop(labels=['publish_time', 'content', 'date'], axis=1)
name = "GLM3-5json"
pred20 = pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRST'))
# ABCDEFGHIJKLMNOPQRST
features = pred20.columns
feature_length = len(features)
data_length = len(data)
# Run 20 GLM question-answer rounds
for i in range(0, feature_length):
    print("No." + str(i + 1) + " query round")
    for j in range(0, data_length):
        background = "China Petroleum & Chemical Corporation (Sinopec) mainly produces crude oil, natural gas, synthetic fibers, fertilizers, rubber, and refined petroleum products. Its competitors include other large state-owned oil and chemical companies and international energy and chemical giants. The impact of financial news on the stock market is complex and diverse. Positive, negative, or neutral information in the news can trigger investor sentiment fluctuations.\
        Positive news can increase investor confidence, making them more inclined to buy stocks and thereby pushing stock prices up;\
        Negative news can trigger panic sentiment, causing investors to sell stocks and leading to falling prices;\
        Neutral news has no obvious positive or negative sentiment tendency and is viewed by investors as normal market conditions or temporary calm. Investors use this information as reference for decisions. In such cases, investors often remain cautious, maintain existing positions, or adopt prudent strategies rather than rush to buy or sell in large quantities, which helps keep the market relatively stable.\
        At the same time, financing-related buy and sell messages are important in financial markets.\
        When financing buy volume is large, used in ways favorable to company development, or market sentiment is optimistic, it is usually seen as positive news and may push stock prices up.\
        Conversely, if financing buy volume is small, used in ways unfavorable to company development, or market sentiment is pessimistic, it may be viewed as negative news and lead to falling prices.\
        When financing buy volume is moderate, usage is ordinary, or the overall market trend is stable, financing buy and sell activity may be viewed as neutral news, and stock-price movements tend to be steadier.\
        When financing sell volume is large, usage aligns with company strategy, or market sentiment is optimistic, it may be viewed as positive news because it can imply the company obtained funds for expansion or investment, boosting investor confidence and pushing prices up.\
        Conversely, if financing sell volume is small, usage is unclear, or market sentiment is pessimistic, it may be viewed as negative news, raising investor concerns about the company’s future development and causing prices to fall.\
        When financing sell volume is moderate, usage is ordinary, or the overall market trend is stable, financing sell activity may be seen as neutral news and stock-price fluctuations are relatively stable."
        content = "Background knowledge：" + background + "Here are three samples：\
        News content:'''On March 25, Sinopec (600028) held its 2023 annual performance briefing. Chairman Ma Yongsheng responded to the decline in 2023 net profit, noting that international crude oil prices fluctuated downward overall in 2023, down 18.4% year-on-year, while the domestic chemical market was oversupplied and margins remained low. Facing these challenges, the company optimized operations across the board, strengthened coordination for efficiency gains, tightened cost control, and improved quality and efficiency across the whole value chain, achieving hard-won results. Ma Yongsheng stated that the year-on-year decline in net profit was mainly due to inventory-change effects and accrued mining-right transfer income. Excluding these factors, profit on a comparable basis improved significantly.''', Analysis:\
        {\
        Entity:Sinopec\
        Event:On March 25, Sinopec held its 2023 annual performance briefing, and Chairman Ma Yongsheng addressed the decline in 2023 net profit.\
        Impact:Although net profit declined year-on-year, Ma Yongsheng explained the reasons and highlighted achievements in operational optimization and cost control. Excluding certain unfavorable factors, comparable-basis profit improved noticeably.\
        Sentiment：neutral\
        }；\
       News content: '''According to Securities Star, Sinopec’s 2023 annual report showed operating revenue of 3,212.215 billion yuan, down 3.19% year-on-year; net profit attributable to parent of 60.463 billion yuan, down 9.87% year-on-year; and non-recurring-adjusted net profit of 60.234 billion yuan, up 3.92% year-on-year. In Q4 2023, quarterly operating revenue was 742.274 billion yuan, down 14.17%; quarterly net profit attributable to parent was 7.497 billion yuan, down 23.79%; and quarterly non-recurring-adjusted net profit was 9.955 billion yuan, up 417.14%. Debt ratio was 52.7%, investment income 5.811 billion yuan, financial expenses 9.922 billion yuan, and gross margin 15.65%.''', Analysis:\
        {\
        Entity： Sinopec\
        Event: Sinopec released its 2023 annual report, showing year-on-year declines in operating revenue and attributable net profit, while non-recurring-adjusted net profit increased year-on-year; in Q4, operating revenue and attributable net profit declined year-on-year, while non-recurring-adjusted net profit rose sharply.\
        Impact: Although non-recurring-adjusted net profit increased, declines in operating revenue and attributable net profit indicate operating pressure and challenges. In addition, a relatively high debt ratio may negatively affect the company’s future financial condition. Overall, the annual report reflects operational difficulties under the current market environment.\
        Sentiment： negative\
        }；\
        News content:'''On March 13, Sinopec and Contemporary Amperex Technology Co., Limited (CATL) signed a strategic cooperation framework agreement in Beijing. Both parties stated they will further deepen strategic cooperation, broaden cooperation areas, extend the industrial chain, accelerate transformation and upgrading, and bring cooperation to a new level.''', Analysis:\
        {\
        Entity： Sinopec\
        Event: Sinopec and CATL signed a strategic cooperation framework agreement in Beijing.\
        Impact: Signing this agreement indicates deeper strategic cooperation, broader cooperation fields, extended industrial chains, and faster transformation and upgrading, which is expected to drive greater development and innovation in both companies’ respective fields, and enhance market competitiveness and investor confidence.\
        Sentiment： positive\
        }。\
        Please act as an expert in financial-news sentiment classification. Refer to the sample cases above and the background knowledge to analyze the sentiment conveyed by the news content in triple quotes below using JSON fields (Entity, Event, Impact, Sentiment). Determine whether it is positive, negative, or neutral. If the sentiment cannot be determined, default to neutral. Output JSON in the following format:\
        {\
        Entity:Sinopec\
        Event：\
        Impact：\
        Sentiment：\
        }\
        News content:'''" + data["news_full_text"][j] + "''', Analysis:"

        response, history = model.chat(tokenizer, content, history=[])
        # Use regex to match the "sentiment" field
        emotion = re.search(r'Sentiment：(.+)', response)
        if emotion == None:
            emotion = re.search(r'Sentiment:(.+)', response)
        if emotion != None:
            pred20.loc[j, features[i]] = emotion.group(1).strip()
        else:
            pred20.loc[j, features[i]] = "neutral"
        if j % 40 == 0:
            print("No." + str(j) + " records")
            print(response, emotion)

# filePath="/kaggle/input/full-text-news/_.csv"
# df=pd.read_csv(filePath,encoding="UTF-8")
# Example: drop metadata columns before inference.
# pred20=pd.read_csv('/kaggle/input/pred-label/pred.csv',encoding='utf-8')

# import sys
sys.path.append("../output")
pred20 = pred20.reset_index(drop=True)
pred20.to_csv('/kaggle/working/' + name + '.csv', index_label="item")
# print("label")


for i in range(0,data_length):
    counts = {"positive": 0, "negative": 0, "neutral": 0}  # Initialize counters
    for value in pred20.iloc[i, :-1]:
        value=value[:2]
#         print(value)
        if value== "positive" or value=="rise"  or value=="positive":
            counts["positive"] += 1
        elif value == "negative" or value=="fall":
            counts["negative"] += 1
        elif value == "neutral" or value=="financing" or value=="unable" or value=="stable":
            counts["neutral"] += 1
    if counts["positive"]>counts["negative"] and counts["positive"]>counts["neutral"]:
        pred20.loc[i,"label"]="positive"
    elif counts["negative"]>counts["positive"] and counts["negative"]>counts["neutral"]:
        pred20.loc[i,"label"]="negative"
    else :
        pred20.loc[i,"label"]="neutral"
predicted_labels=pred20["label"]
test_y=data["label"]
# Calculate accuracy
accuracy=np.sum(data["label"]==pred20["label"])/data_length
# Calculate F1 score
f1_micro = f1_score(test_y, predicted_labels, average='weighted')
# Calculate recall
recall = recall_score(test_y, predicted_labels, average='weighted')  # You can choose other average settings.
# Calculate precision
precision = precision_score(test_y, predicted_labels, average='weighted')  # You can choose other average settings.
print(f"Recall: {recall:.4f}\nPrecision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}\nF1 score：{f1_micro:.4f}")

true=[]
pred=[]
for i in range(data_length):
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
# Compute the number of true samples in each row
row_sums = cm.sum(axis=1, keepdims=True)
# Divide each confusion-matrix cell by its row total to obtain probabilities.
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/news_full_text_labeled.png")
plt.show()
