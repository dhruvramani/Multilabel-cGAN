import os
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_epoch, batch_size = 100, 50
input_dim, random_dim, output_dim = 500, 100, 983
x = (Input(shape = (input_dim, ), dtype='float32'))
noise = (Input(shape=(random_dim, ), dtype='float32'))
y = (Input(shape = (output_dim, ), dtype='float32'))

def get_data(path, noise=False):
    data = np.load(path)
    if noise == True :
        data = data + np.random.normal(0, 0.001, data.shape)
    return data

def generator_model():
    start = concatenate([noise, x])
    layer1 = Dense(350, input_dim = input_dim + random_dim, activation='relu')(start)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(120, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer3 = Dense(460, activation='relu')(layer2)
    layer3 = Dropout(0.5)(layer3)
    layer4 = Dense(780, activation='relu')(layer3)
    layer4 = Dropout(0.5)(layer4)
    layer5 = Dense(output_dim, activation='sigmoid')(layer4)
    model = Model(input=[noise, x], output=layer5)
    return model

def discriminator_model(): # Concatenated
    start = concatenate([])
    layer1 = Dense(680, input_dim = output_dim + input_dim, activation='relu')(start)
    layer1 = Dropout(0.5)(layer1)
    layer2 = Dense(350, activation='relu')(layer1)
    layer2 = Dropout(0.5)(layer2)
    layer3 = Dense(100, activation='relu')(layer2)
    layer3 = Dropout(0.5)(layer3)
    layer4 = Dense(1, activation='sigmoid')(layer3)
    model = Model(start, output=layer4)
    return model

def generator_containing_discriminator(generator, discriminator):
    fake = generator([x, noise])
    discriminator.trainable = False
    disc_val = discriminator([x, fake])
    model = Model(input=[x, noise], output=disc_val)

def train():
    X_train, Y_train = get_data('./data/delicious/delicious-train-features.pkl'), get_data('./data/delicious/delicious-train-labels.pkl')
    discriminator = discriminator_model()
    generator = generator_model()
    disc_on_gen = generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer='SGD')
    disc_on_gen.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, random_dim))
    for epoch in range(num_epoch):
        num_batches = int(X_train.shape[0] / batch_size)
        print("Epoch : {}, Number of Epochs : {}".format(epoch, num_batches))

        for index in range(num_batches):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, random_dim)
            x_batch, y_batch = X_train[index * batch_size : (index + 1) * batch_size], Y_train[index * batch_size : (index + 1) * batch_size]
            fake_y = generator.predict([x_batch, noise], verbose=0)
            real_pairs = np.concatenate((x_batch, y_batch), axis=1)
            fake_pairs = np.concatenate((x_batch, fake_y), axis=1)
            x = np.concatenate((real_pairs, fake_pairs))
            y = np.asarray([1] * batch_size + [0] * batch_size, dtype=np.float32)
            d_loss = discriminator.train_on_batch(x, y)
            print("Batch {} -  Disc. Loss : {} ".format(index, d_loss), end="")
            discriminator.trainable = False
            g_loss = disc_on_gen.train_on_batch([x_batch, noise], np.asarray([1] * batch_size, dtype=np.float32))
            discriminator.trainable = True
            print(" Gen. Loss : {}".format(index, d_loss), end="\r")
            
            if index % 10 == 9:
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
