from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import keras
from keras.models import load_model
import os
import numpy as np
import spacy
import pandas as pd
import keras.losses
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import keras.backend as K
from sklearn.preprocessing import MultiLabelBinarizer
from utils import preprocess_text
from semantic_search import searchresults
EN = spacy.load('en_core_web_sm')

# Flask App stuff
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Custom loss function to handle multilabel classification task
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

MAX_SEQUENCE_LENGTH = 300
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
keras.losses.multitask_loss = multitask_loss
global graph
graph = tf.compat.v1.get_default_graph()

# Flask API Routes
@app.route('/')
@cross_origin()
def homepage():
    return jsonify({'test': "Working!"})

@app.route('/getsearchresults')
@cross_origin()
def getsearchresults():
    params = request.json
    if (params == None):
        params = request.args

    query = params["query"]
    query = preprocess_text(query)
    results = searchresults(query, params["num_results"])
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)