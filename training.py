import pandas as pd 
from gensim.models import word2vec as wv
import numpy as np 
import nltk 

dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(["Unnamed: 0"])

#Tokenize words
corpus = []
for words in dataset.comment:
    words = words.split()
    corpus.append(words)

#Training word 2vec 
vocab_size = 300
min_counts = 6
down_sample = 1e-2
context =   3
vectors = wv.Word2Vec(corpus,
                     size = vocab_size,
                     workers =4,
                     window = context,
                     min_count = min_counts,
                     sample = down_sample)


vectors.most_similar("lora")
wordset = set(vectors.wv.index2word) #Checks if the word is in the Word2vec corpus 

allVectors = np.zeros(300,dtype="object")
counter = 0 
vec = []

for words in corpus:    
    featureVec = np.zeros(300,dtype="object")
    for word in words:
        if word in wordset:
            featureVec = np.add(featureVec,vectors[word])
    vec.append(featureVec.T)
    counter = counter + 1
    print(counter)


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

train_X, X_test, train_Y, y_test = train_test_split(vec, dataset.rating, test_size = 0.20, random_state = 123)
classifier = SGDClassifier()    
classifier.fit(train_X,train_Y)
y_pred = classifier.predict(X_test)

forest = RandomForestClassifier(n_estimators = 5)
forest.fit(train_X, train_Y)
y_pred = forest.predict(X_test)
cm = confusion_matrix(y_pred,y_test)
print(accuracy_score(y_test, y_pred))