
import numpy as np
import pandas as pd
import re
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score

df= pd.read_csv('SMSSpamCollection.tsv',sep="\t",names=['label','text'])

sn=SnowballStemmer('english')
stop = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = nltk.word_tokenize(text)
    text=[t for t in text if len(t)>1 ]
    text=[sn.stem(word) for word in text if word not in stop ]
    text= ' '.join(text)
    return text

df['clean_text']=df['text'].apply(clean_text)

hamdata= df[df['label']=='ham']
hamdata=hamdata['clean_text']


def wordCloud(text):
    words = ' '.join(text)
    wordcloud = WordCloud().generate(words)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#wordCloud(hamdata)
spamdata= df[df['label']=='spam']
spamdata=spamdata['clean_text']
#wordCloud(spamdata)
cv = TfidfVectorizer(max_features=5000)
X= cv.fit_transform(df['clean_text']).toarray()

y=pd.get_dummies(df['label'])
y=y['spam'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = MultinomialNB().fit(X_train, y_train)
y_prd= model.predict(X_test)
acc=model.score(X_train, y_train)
print("Accuracy in training data\n" + str(acc))
acct=model.score(X_test, y_test)
print("Accuracy in test data\n" + str(acct))


print(confusion_matrix(y_test, y_prd))

print(classification_report(y_test, y_prd))
