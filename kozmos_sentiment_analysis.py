from warnings import filterwarnings
import matplotlib.pyplot as plt
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
drops = temp_df[temp_df < 1000]
df["REVIEW"] = df["REVIEW"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization
df["REVIEW"] = df["REVIEW"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))