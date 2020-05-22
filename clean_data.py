import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

col_list = ["title", "text", "subject"]
fake = pd.read_csv('Fake.csv', usecols=col_list, encoding="UTF-8")
real = pd.read_csv('True.csv', usecols=col_list, encoding="UTF-8")

fake['label'] = "fake"
real['label'] = "real"

# create a dataframe using texts and lables
df = pd.DataFrame()
df = df.append(fake)
df = df.append(real)

#clean data
pd.set_option('display.max_colwidth', -1)
df['title'] = df['title'].str.lower()
df['text'] = df['text'].str.lower()

df['title'] = df['title'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
df['text'] = df['text'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)

stemmer = SnowballStemmer('english')
stop = stopwords.words('english')
df['title'] = df['title'].apply(lambda x: [stemmer.stem(item) for item in str(x).split() if item not in stop])
df['text'] = df['text'].apply(lambda x: [stemmer.stem(item) for item in str(x).split() if item not in stop])

#shuffle data frames
df = df.sample(frac=1)

df.to_csv("news_cleaned.csv", index=False)













