import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, Adagrad
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_epoch, batch_size = 100, 50
input_dim, random_dim, output_dim = 500, 100, 983 # For 'Delicious' dataset
x = (Input(shape = (input_dim, ), dtype='float32'))
noise = (Input(shape=(random_dim, ), dtype='float32'))
y = (Input(shape = (output_dim, ), dtype='float32'))

def get_data(path, noise=False):
    data = np.load(path)
    if noise == True :
        data = data + np.random.normal(0, 0.001, data.shape)
    return data

def generator_model():
    global input_dim
    global output_dim
    global random_dim
    global noise
    global x
    global y
    start = concatenate([x, noise])
    layer1 = Dense(350, input_dim = input_dim + random_dim, activation='relu')(start)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(120, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer3 = Dense(460, activation='relu')(layer2)
    layer3 = Dropout(0.5)(layer3)
    layer4 = Dense(780, activation='relu')(layer3)
    layer4 = Dropout(0.5)(layer4)
    layer5 = Dense(output_dim, activation='sigmoid')(layer4)
    model = Model(input=[x, noise], output=layer5)
    return model

def discriminator_model(): # Concatenated
    global input_dim
    global output_dim
    global random_dim
    inp = (Input(shape = (input_dim + output_dim, ), dtype='float32'))
    layer1 = Dense(680, input_dim = output_dim + input_dim, activation='relu')(inp)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(350, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer3 = Dense(100, activation='relu')(layer2)
    layer3 = Dropout(0.5)(layer3)
    layer4 = Dense(1, activation='sigmoid')(layer3)
    model = Model(input=inp, output=layer4)
    return model

def generator_containing_discriminator(generator, discriminator):
    global noise
    global x
    global y
    fake = generator([x, noise])
    discriminator.trainable = False
    disc_input = concatenate([x, fake])
    disc_val = discriminator(disc_input)
    model = Model(input=[x, noise], output=[fake, disc_val])
    return model

def discriminator_loss(y_true, y_pred):
    global batch_size
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.concatenate([K.ones_like(K.flatten(y_pred[:batch_size,:])), K.zeros_like(K.flatten(y_pred[:batch_size,:])) ]) ), axis=-1)

def discriminator_on_generator_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.ones_like(K.flatten(y_pred))), axis=-1)

def generator_l1_loss(y_true,y_pred):
    return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)

def train():
    global random_dim
    global num_epoch
    global batch_size
    X_train, Y_train = get_data('./data/delicious/delicious-train-features.pkl'), get_data('./data/delicious/delicious-train-labels.pkl')
    discriminator = discriminator_model()
    generator = generator_model()
    disc_on_gen = generator_containing_discriminator(generator, discriminator)
    generator.compile(loss='binary_crossentropy', optimizer='SGD')
    disc_on_gen.compile(loss=[generator_l1_loss, discriminator_on_generator_loss], optimizer="rmsprop")
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_loss, optimizer="rmsprop")
    noise = np.zeros((batch_size, random_dim))
    for epoch in range(num_epoch):
        num_batches = int(X_train.shape[0] / batch_size)
        print("Epoch : {}".format(epoch))
        for index in range(num_batches):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, random_dim)
            x_batch, y_batch = X_train[index * batch_size : (index + 1) * batch_size], Y_train[index * batch_size : (index + 1) * batch_size]
            fake_y = generator.predict([x_batch, noise], verbose=0)
            real_pairs = np.concatenate((x_batch, y_batch), axis=1)
            fake_pairs = np.concatenate((x_batch, fake_y), axis=1)
            X = np.concatenate((real_pairs, fake_pairs))
            Y = np.asarray([1] * batch_size + [0] * batch_size, dtype=np.float32)
            d_loss = discriminator.train_on_batch(X, Y)
            print("Batch {} -  Disc. Loss : {} ".format(index, d_loss), end="")
            discriminator.trainable = False
            g_loss = disc_on_gen.train_on_batch([x_batch, noise], [y_batch ,np.asarray([1] * batch_size, dtype=np.float32)])
            discriminator.trainable = True
            print(" Gen. Loss : {}".format(g_loss[1]), end="\r\r")
            
            if(int(index % 100) == 9):
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)

def test(data_type='test'):
    X, Y = get_data('./data/delicious/delicious-{}-features.pkl'.format(data_type)), get_data('./data/delicious/delicious-{}-labels.pkl'.format(data_type))
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    gen_y = generator.predict(X)

if __name__ == '__main__':
    train()
    test()
