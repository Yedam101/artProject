import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm, tqdm_notebook
import random

import imageio
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import img_to_array

import io
import base64
from markupsafe import Markup

from numpy.random import seed
seed(1)
tf.random.set_seed(1)

### Basic global variables

artists = pd.read_csv('./root/artists.csv')
artists = artists.sort_values(by=['paintings'], ascending=False)
artists_top = artists[artists['paintings'] >= 200].reset_index()
artists_top = artists_top[['name', 'paintings']]


hund = [100 for i in range(10)]
hund = np.array(hund)

labels = list(artists_top['name'].values)
train_input_shape = (224, 224, 3)
n_classes=10

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=train_input_shape)
X = base_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(16, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_classes, activation='softmax')(X)

PredictionModel = Model(inputs=base_model.input, outputs=output)
PredictionModel.load_weights('./root/model_weights.hdf5')

# ## Deep learning prediction model


def ArtistPrediction(url):
    web_image = imageio.imread(url)
    web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
    web_image = np.array(web_image, dtype='f')
    web_image = web_image / 255
    web_image = np.expand_dims(web_image, axis=0)

    prediction = PredictionModel.predict(web_image)
    prediction_probability = np.array(prediction)[0]
    prediction_probability = np.sort(prediction_probability)[::-1]
    prediction_prob = prediction_probability*hund
    prediction_idx = np.argsort(prediction)[0][::-1]

    prediction_idx_list = []
    for i in prediction_idx:
        prediction_idx_list.append(labels[i])

    answer_labels = []
    prediction_prob_list = []

    for i in range(3):
        answer_labels.append(labels[prediction_idx[i]].replace('_', ' '))
        prediction_prob_list.append(prediction_prob[i])
        
    ans_prob = []    
    for i in range(3):
        add = str(answer_labels[i]) + '\n' + str(round(prediction_prob_list[i],2))
        ans_prob.append(add)

    return answer_labels, prediction_prob_list, ans_prob


from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/", methods=["GET"])
    def index():
        return render_template('index.html')

    @app.route("/model", methods=["GET"])
    def model():
        url = request.args.get('url')
        answer_labels, prediction_prob_list, ans_prob = ArtistPrediction(url)

        # top1 artist name
        artist_name = answer_labels[0]

        # Figure 1. Show original url figure
        s = io.BytesIO()

        plt.imshow(imageio.imread(url))
        plt.axis('off')
        plt.savefig(s, format='png', bbox_inches="tight")
        plt.close()

        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        Figure1 = Markup(f'<img align="left" src="data:image/png;base64,%s">' % s)

        # Figure 2. Show bar-plot (probability)
        prob = pd.Series(prediction_prob_list)
        label_freq = ans_prob

        s = io.BytesIO()
        fig = plt.figure(figsize= (10,5))
        ax = plt.subplot(1,1,1)
        ax = sns.barplot(y=label_freq, x=prob, order= label_freq, alpha=0.6)

        ax = plt.xlabel("")
        ax = plt.xticks(fontsize= 13, x=-0.03, y=-0.05)
        ax = plt.yticks(fontsize= 14, x=-0.01)
        ax = plt.title("Label frequency", fontsize= 20, x=0.453, y=1.1)
        plt.savefig(s, format='png', bbox_inches="tight")
        plt.close()

        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        Figure2 = Markup(f'<img align="left" src="data:image/png;base64,%s">' % s)
        return render_template('about.html', Figure1=Figure1, Figure2=Figure2, artist_name=artist_name)
        

    return app

