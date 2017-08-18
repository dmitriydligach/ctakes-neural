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

# model parameters
EMBED_DIM = 300
NUM_FILTERS = 200
FILTER_SIZE = 5
DROPOUT_RATE = 0.25
HIDDEN_UNITS = 300
REGUL_COEF = 0.001
LEARN_RATE = 0.0005
NUM_EPOCHS = 8
BATCH_SIZE = 50

def get_model(vocab_size, max_seq_len, init_vectors, num_classes):
  """Get model definition"""

  model = Sequential()
  model.add(Embedding(input_dim=vocab_size,
                      output_dim=EMBED_DIM,
                      input_length=max_seq_len,
                      trainable=True,
                      weights=init_vectors))
  model.add(Conv1D(filters=NUM_FILTERS,
                   kernel_size=FILTER_SIZE,
                   activation='relu'))
  model.add(GlobalMaxPooling1D())

  model.add(Dropout(DROPOUT_RATE))
  model.add(Dense(HIDDEN_UNITS, kernel_regularizer=regularizers.l2(REGUL_COEF)))
  model.add(Activation('relu'))

  model.add(Dropout(DROPOUT_RATE))
  model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(REGUL_COEF)))
  model.add(Activation('softmax'))

  return model

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

    model = get_model(len(provider.word2int), maxlen, init_vectors, classes)
    optimizer = RMSprop(lr=LEARN_RATE, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=0,
              validation_split=0.0)

    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
