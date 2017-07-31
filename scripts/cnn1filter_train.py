#!/usr/bin/env python

import numpy as np
np.random.seed(1337)
import lib.et_cleartk_io as ctk_io
import lib.nn_models
import sys
import os.path
import dataset, word2vec
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras import regularizers
import pickle

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)
    working_dir = args[0]
    data_file = os.path.join(working_dir, 'training-data.liblinear')

    # learn alphabet from training data
    provider = dataset.DatasetProvider(data_file)
    # now load training examples and labels
    train_x, train_y = provider.load(data_file)
    # turn x and y into numpy array among other things
    maxlen = max([len(seq) for seq in train_x])
    classes = len(set(train_y))

    train_x = pad_sequences(train_x, maxlen=maxlen)
    train_y = to_categorical(np.array(train_y), classes)

    pickle.dump(maxlen, open(os.path.join(working_dir, 'maxlen.p'),"wb"))
    pickle.dump(provider.word2int, open(os.path.join(working_dir, 'word2int.p'),"wb"))
    pickle.dump(provider.label2int, open(os.path.join(working_dir, 'label2int.p'),"wb"))

    w2v = word2vec.Model('/home/dima/Data/Word2VecModels/mimic.txt')
    init_vectors = [w2v.select_vectors(provider.word2int)]

    print 'train_x shape:', train_x.shape
    print 'train_y shape:', train_y.shape

    model = Sequential()
    model.add(Embedding(len(provider.word2int),
                        300,
                        input_length=maxlen,
                        trainable=True,
                        weights=init_vectors))
    model.add(Conv1D(filters=200,
                     kernel_size=5,
                     activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dropout(0.25))
    model.add(Dense(300, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))
    model.add(Dense(classes, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=4,
              batch_size=50,
              verbose=0,
              validation_split=0.0)

    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
