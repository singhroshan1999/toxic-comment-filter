import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
DATA_SIZE = 100
def stemmer(data):
    stem = PorterStemmer()
    corpus = []
    count = 0
    for i in data:
        if count%(data.size/50) == 0:
            print(count)
        count+=1
        post = re.sub('[^a-zA-Z]',' ',i)
        post = post.lower()
        post =  post.split()
        post = [stem.stem(word) for word in post if not word in set(stopwords.words('english'))]
        post = ' '.join(post)
        corpus.append(post)
    return corpus
def stemmer_s(strin):
    stem = PorterStemmer()
    corpus = []
    post = re.sub('[^a-zA-Z]',' ',strin)
    post = post.lower()
    post =  post.split()
    post = [stem.stem(word) for word in post if not word in set(stopwords.words('english'))]
    post = ' '.join(post)
    corpus.append(post)
    return corpus
def save(mdlName,classifier,count_vector):
    clsf = open(mdlName+'.mdl','wb')
    cnt = open(mdlName+'_count_vector.mdl','wb')
    pickle.dump(classifier,clsf)
    pickle.dump(count_vector,cnt)
def train(dataset,do_save = False):
    corpus = stemmer(dataset['comment_text'][:DATA_SIZE])
    count_vector = CountVectorizer(max_features = 40000)
    x = count_vector.fit_transform(corpus).toarray()
    y = []
    for i in dataset.iloc[:DATA_SIZE,-1].values:
        if i > 0:
            y.append(1)
        else:
            y.append(0)
    classifier = RandomForestClassifier(n_estimators= 100,criterion = 'gini',random_state = 0)
    classifier.fit(x,y)
    if do_save:
        import random
        filename = 'toxic'+str(random.randint(1000,9999))
        save(filename,classifier,count_vector)
    return classifier,count_vector
def load(mdlName):
    clsf = open(mdlName+'.mdl','rb')
    cnt = open(mdlName+'_count_vector.mdl','rb')
    return pickle.load(clsf),pickle.load(cnt)
def isToxic(s,classifier,count_vector):
     return (classifier.predict(count_vector.transform(stemmer_s(s)))[0] != 0)
def show(dataset):
    data_low = dataset
    corpus_low = stemmer(data_low['comment_text'][:DATA_SIZE])
    count_vector = CountVectorizer(max_features = 40000)
    x = count_vector.fit_transform(corpus_low).toarray()
    y = []
    for i in data_low.iloc[:DATA_SIZE,-1].values:
        if i > 0:
            y.append(1)
        else:
            y.append(0)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
    classifier = RandomForestClassifier(n_estimators= 100,criterion = 'gini',random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    c = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            print(y_pred[i],y_test[i],x_test[i])
            c+=1
            print(c)
    return classifier,count_vector
if __name__ == "__main__":
    print("this is toxic comment module not runnable")
