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
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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

## Binarizing
df["SENTIMENT_LABEL"] = df["REVIEW"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["SENTIMENT_LABEL"] = LabelEncoder().fit_transform(df["SENTIMENT_LABEL"])

## Splitting

X_train, X_test, y_train, y_test = train_test_split(df["REVIEW"], df["SENTIMENT_LABEL"],
                                                    test_size=0.2, stratify=df.SENTIMENT_LABEL,
                                                    random_state=26)
## Text Vectorizing

tf_idf_word_vectorizer = TfidfVectorizer()
X_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_train)
X_test_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_test)

## Modelling (Recheck, maybe there is data leakage somewhere. It looks "unbelievably good")
from sklearn.metrics import classification_report

lr = LogisticRegression(random_state=26, class_weight="balanced").fit(X_train_tf_idf_word, y_train)

y_pred = lr.predict(X_test_tf_idf_word)

report = classification_report(y_test, y_pred)
print(report)


fig = plt.figure(figsize=(9, 9))
g = fig.add_subplot(1,1,1)
plot_confusion_matrix(lr, X_test_tf_idf_word, y_test, ax=g)
plt.show()