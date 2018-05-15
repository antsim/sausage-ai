# encoding=utf-8
"""
    Inspired by https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    Run with: python main.py ../data/cannibalcorpse.txt 400
"""

from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import xml.etree.ElementTree as ET
import helper
import numpy as np
import random
import sys
import tensorflow as tf
from pathlib import Path
from keras.models import load_model
from keras.callbacks import Callback
from bisect import bisect_left
from os.path import splitext
from os.path import basename

tf.logging.set_verbosity(tf.logging.ERROR)

"""
    Define global variables.
"""
SEQUENCE_LENGTH = int(sys.argv[2])
SEQUENCE_STEP = 3
PATH_TO_CORPUS = sys.argv[1]
EPOCHS = 60
DIVERSITY = 1.0

print(sys.argv)
corpus = helper.read_corpus(PATH_TO_CORPUS)
sentence = corpus[0:SEQUENCE_LENGTH]
SENTENCE = sentence

print("using sentence: " + sentence)

def SaveModel():
    model.save(corpusFile[0]+".h5")

def PrintResults():
    for diversity in [0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = SENTENCE
        sentence = sentence.lower()
        generated += sentence

        for i in range(400):
            x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_to_index[char]] = 1.

            predictions = model.predict(x, verbose=0)[0]
            next_index = helper.sample(predictions, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        
        print()
        print()

class TestCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        PrintResults()
        SaveModel()

"""
    Read the corpus and get unique characters from the corpus.
"""
text = helper.read_corpus(PATH_TO_CORPUS)
chars = helper.extract_characters(text)

"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
sequences, next_chars = helper.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)
char_to_index, indices_char = helper.get_chars_index_dicts(chars)

"""
    The network is not able to work with characters and strings, we need to vectorise.
"""
X, y = helper.vectorize(sequences, SEQUENCE_LENGTH, chars, char_to_index, next_chars)

"""
    Define the structure of the model.
"""
corpusFile = splitext(basename(Path(PATH_TO_CORPUS)))
modelFile = Path(corpusFile[0]+".h5")
if modelFile.is_file():
    model = load_model(corpusFile[0]+".h5")
    model.summary()
    PrintResults()
else:
    model = helper.build_model(SEQUENCE_LENGTH, chars)
    model.summary()

model.fit(X, y, 128, EPOCHS, callbacks=[TestCallback()])