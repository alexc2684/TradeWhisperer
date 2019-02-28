import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class USEClassifier:

    def __init__(self, epochs, batch_size, embed_size, model_name, embedding):
        self.embed_size = embed_size
        self.embedding = embedding #TODO: dictionary of embeddings to choose from
        self.model_name = 'test'
        self.batch_size = 4
        self.epochs = epochs

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        #module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed_module = hub.Module(module_url)

        self.model = self.init_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'acc'])

    def init_model(self):
        input_text = Input(shape=(1,), dtype=tf.string)
        embedding = Lambda(self._UniversalEmbedding, output_shape=(self.embed_size,))(input_text)
        dense = Dense(256, activation='relu', kernel_regularizer=l2(.01))(embedding)
        batch_norm = BatchNormalization()(dense)
        dropout = Dropout(.2)(batch_norm)
        pred = Dense(3, activation='softmax', kernel_regularizer=l2(.005))(dropout)
        model = Model(inputs=[input_text], outputs=pred)
        return model

    def train(self, train_X, train_y, test_X, test_y):
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                TensorBoard(log_dir='logs/{}'.format(self.model_name), batch_size=self.batch_size, write_images=True),
                ModelCheckpoint(filepath=self.model_name + '.h5', save_best_only=True)]
        self.model.fit(train_X,
                train_y,
                validation_data=(test_X, test_y),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=2)

    def predict(self, data):
        return self.model.predict(data)

    def _UniversalEmbedding(self, x):
        return self.embed_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

