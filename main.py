from src.neuro_fuzzy_lstms import * 
from src.neuro_fuzzy import * 
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist 
from sklearn.metrics import accuracy_score
import time  
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf 


def load_mnist():
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0],-1)) / 255. 
    X_test = X_test.reshape((X_test.shape[0],-1)) / 255.
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
    return X_train,y_train,X_val,y_val


def with_lstm():
    X_train, y_train, X_test, y_test = load_mnist()

    max_neurons = 10
    threshold_epochs = 3
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, -1))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, -1))

    architecture_with = NeuroFuzzyNetwork(
        X_train_lstm.shape[2], y_train.shape[1], 10, 5, max_neurons, threshold_epochs)
    start_time = time.time()
    history_with_lstm = architecture_with.train(
        X_train_lstm, y_train, epochs=10)
    end_time = time.time()
    pred = architecture_with.predict(X_test_lstm)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    return architecture_with, acc, end_time - start_time, history_with_lstm


def without_lstm():
    X_train, y_train, X_test, y_test = load_mnist()

    max_neurons = 10
    threshold_epochs = 3
    X_train_no_lstm = X_train.reshape((X_train.shape[0], -1))
    X_test_no_lstm = X_test.reshape((X_test.shape[0], -1))

    architecture_without = NeuroFuzzyNetwork_(
        X_train_no_lstm.shape[1], y_train.shape[1], 10, 5,max_neurons, threshold_epochs)
    history_without_lstm = architecture_without.train(
        X_train_no_lstm, y_train, epochs=10)

    start_time = time.time()
    pred = architecture_without.predict(X_test_no_lstm)
    end_time = time.time()

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    return architecture_without, acc, end_time - start_time, history_without_lstm

def normal():
    X_train,y_train,X_test,y_test = load_mnist()
    model = Sequential()
    model.add(Dense(10,activation='relu',input_shape=(784,)))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(X_train,y_train,epochs=10,batch_size=32,verbose=0)
    start_time = time.time()
    pred = model.predict(X_test)
    end_time = time.time()
    acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(pred,axis=1))
    return model,acc,end_time-start_time,history



