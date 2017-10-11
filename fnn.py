import os
import numpy as np
import eval_performance
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_data(path, noise=False):
    data = np.load(path)
    if noise == True :
        data = data + np.random.normal(0, 0.001, data.shape)
    return data

num_epoch, batch_size = 100, 50
input_dim, random_dim, output_dim = 500, 100, 983
model = Sequential()
model.add(Dense(600, input_dim=input_dim, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(output_dim, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics=[eval_performance.patk])
X_train, X_test, Y_train, Y_test = get_data("./data/delicious/delicious-train-features.pkl"), get_data("./data/delicious/delicious-test-features.pkl"), get_data("./data/delicious/delicious-train-labels.pkl"), get_data("./data/delicious/delicious-test-labels.pkl")
model.fit(X_train, Y_train, epochs=num_epoch, batch_size = batch_size)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size = batch_size)
classes = model.predict(X_test, batch_size=batch_size)
print(loss_and_metrics)