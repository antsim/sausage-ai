"""

  Generate lyrics based on the trained model
  Run with python lyrgen.py MODEL_FILE CORPUS_FILE DIVERSITY
  eg. python lyrgen.py yo.h5 ../data/yo.txt 1.2

"""
from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import helper
import sys

import numpy as np
import tensorflow as tf

from pathlib import Path
from os.path import splitext
from os.path import basename
from keras.models import load_model

tf.logging.set_verbosity(tf.logging.ERROR)

CORPUS = helper.read_corpus(sys.argv[2])
PATH_TO_MODEL = sys.argv[1]
DIVERSITY = float(sys.argv[3])
GEN_LENGTH = 400
CHARS = helper.extract_characters(CORPUS)
char_to_index, indices_char = helper.get_chars_index_dicts(CHARS)

"""
  Load the model
"""
modelFile = Path(PATH_TO_MODEL)
if modelFile.is_file():
    model = load_model(PATH_TO_MODEL)

"""
  GEN_LENGTH needs to be the same that was used when model was saved
"""
generated = ''
sentence = CORPUS[0:GEN_LENGTH]
sentence = sentence.lower()
generated += sentence

for z in range(50):
  for i in range(GEN_LENGTH):
    x = np.zeros((1, GEN_LENGTH, len(CHARS)))
    for t, char in enumerate(sentence):
      x[0, t, char_to_index[char]] = 1.

    predictions = model.predict(x, verbose=0)[0]
    next_index = helper.sample(predictions, DIVERSITY)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
  print()