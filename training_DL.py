def getVectors(corpus,vectors,size):
    wordset = set(vectors.wv.index2word) #Checks if the word is in the Word2vec corpus 
    vec = []
    counter = 0
    for words in corpus:    
        featureVec = np.zeros(size,dtype="object")
        for word in words:
            if word in wordset:
                featureVec = np.add(featureVec,vectors[word])
        vec.append(featureVec.T)
        counter = counter + 1
        print(counter)
    return vec


import numpy as np
import pandas as pd 
from gensim.models import word2vec as wv
dataset = pd.read_csv("data.csv")


#Tokenize words
corpus = []
for words in dataset.comment:
    words = words.split()
    corpus.append(words)

from keras.utils import to_categorical
Y = to_categorical(dataset.rating)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(corpus,Y, test_size = 0.20, random_state = 123)

#Training word 2vec 
vocab_size = 500
min_counts = 7
down_sample = 1e-2
context =   2
vectors = wv.Word2Vec(X_train,
                     size = vocab_size,
                     workers =4,
                     window = context,
                     min_count = min_counts,
                     sample = down_sample)



X_train = getVectors(X_train,vectors,vocab_size)
X_test = getVectors(X_test,vectors,vocab_size)
del(corpus)

X_train = np.array(X_train)

from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = vocab_size))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 8)


y_pred = classifier.predict(np.array(X_test))

y_pred = np.argmax(y_pred,axis = 1)
y_test = np.argmax(y_test,axis = 1)


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_pred,y_test)
print(accuracy_score(y_test, y_pred))



test = ["bharwa","acha","acha theek ", "siray"]
n = []
for words in test:
    words = words.split()
    n.append(words)

test = n
test = getVectors(test,vectors,vocab_size)
np.argmax(classifier.predict(np.array(test)),axis = 1)

