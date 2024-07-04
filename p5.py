import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pickle import *

data = pd.read_csv("spam_ap24.csv")
print(data)

print(data.isnull().sum())

ps = PorterStemmer()

def clean_data(txt):
    txt = txt.lower()
    txt = word_tokenize(txt)
    txt = [t for t in txt if t not in punctuation]
    txt = [t for t in txt if t not in stopwords.words("english")]
    txt = [ps.stem(t) for t in txt]
    txt = " ".join(txt)
    return txt

data["Clean_Message"] = data["Message"].apply(clean_data)
print(data)

tv = TfidfVectorizer()
tvector = tv.fit_transform(data["Clean_Message"])

features = pd.DataFrame(tvector.toarray(),columns=tv.get_feature_names_out())
print(features)
target = data["Category"]
print(target)

x_train,x_test,y_train,y_test = train_test_split(features,target)
model = MultinomialNB()
model.fit(x_train,y_train)

f = open("vector.pkl","wb")
dump(tv,f)
f.close

f = open("model2.pkl","wb")
dump(model,f)
f.close
