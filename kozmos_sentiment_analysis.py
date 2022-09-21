from warnings import filterwarnings
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

amazon = pd.read_excel("NLP/datasets/amazon.xlsx")

df = amazon.copy()

df.head()
df.info()

df.columns = [col.upper() for col in df.columns]
df["REVIEW"] = df["REVIEW"].str.lower()

# Getting rid of punctuations and digits
df["REVIEW"] = df["REVIEW"].str.replace('[^\w\s]', '')
df["REVIEW"] = df["REVIEW"].str.replace('\d', '')

# Stop Words
sw = stopwords.words("english")
df["REVIEW"] = df["REVIEW"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rare Words
temp_df = pd.Series(' '.join(df["REVIEW"]).split()).value_counts()
drops = temp_df[temp_df < 200]
df["REVIEW"] = df["REVIEW"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization
df["REVIEW"] = df["REVIEW"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Term Frequencies
tf = df["REVIEW"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

## Visualization
text = " ".join(i for i in df.REVIEW)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

fig = plt.figure(figsize=(9, 6))
g = sns.barplot(data=tf.sort_values("tf", ascending=False)[0:10], x="words", y="tf", palette="viridis")
g.set_title("Word frequencies")
g.set_xlabel("Word")
g.set_ylabel("Count")
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=6)
g.tick_params(which="minor", length=4)
plt.show()

### Sentiment Analysis

sia = SentimentIntensityAnalyzer()

df["REVIEW"][0:10].apply(lambda x: sia.polarity_scores(x))
df["REVIEW"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# df["POLARITY_SCORE"] = df["REVIEW"].apply(lambda x: sia.polarity_scores(x)["compound"])

## Binarizing the labels
df["SENTIMENT_LABEL"] = df["REVIEW"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

