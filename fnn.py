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

def patk(y_true, y_pred):
   score, up_opt = tf.metrics.sparse_precision_at_k(y_true, y_pred, 10)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

num_epoch, batch_size = 75, 50
input_dim, random_dim, output_dim = 500, 100, 983
model = Sequential()
model.add(Dense(600, input_dim=input_dim, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(output_dim, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
X_train, X_test, Y_train, Y_test = get_data("./data/delicious/delicious-train-features.pkl"), get_data("./data/delicious/delicious-test-features.pkl"), get_data("./data/delicious/delicious-train-labels.pkl"), get_data("./data/delicious/delicious-test-labels.pkl")
model.fit(X_train, Y_train, epochs=num_epoch, batch_size = batch_size)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size = batch_size)

num_batches = int(X_train.shape[0] / batch_size)
for index in range(num_batches):
    x_batch, y_batch = X_test[index * batch_size :(index+1) * batch_size, : ], Y_test[index * batch_size : (index + 1), :]
    predictions = model.predict_on_batch(x_batch)
    print(eval_performance.evaluate(y_batch, predictions))