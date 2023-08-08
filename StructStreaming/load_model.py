import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Model
from keras import backend as K
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.layers import Embedding, Reshape, Flatten, Dropout, Concatenate, Conv2D, MaxPool2D
from keras.layers import Dense, Input, Bidirectional, GRU, LSTM
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from pyspark.sql import functions as f
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.tf2 import Estimator
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

OrcaContext.log_output = True
cluster_mode = "local"


if cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=1)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
sequence_length = 100
num_words = len(word_index) + 1
embedding_dim = 300
drop = 0.5
batch_size = 256

with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)


with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)


def model_creator_text_cnn(config):
    import tensorflow as tf
    filter_sizes = [2, 3, 5]
    num_filters = 32

    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim,
                          input_length=sequence_length, weights=[embedding_matrix])(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(
        filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='elu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(
        filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='elu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(
        filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='elu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(
        sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(
        sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(
        sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=3, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    adam = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def model_creator_gru(config):
    import tensorflow as tf

    input = Input(shape=(sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix])(input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output = Dense(3, activation="softmax")(conc)

    # this creates a model that includes
    model = Model(inputs=input, outputs=output)
    adam = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def model_creator_lstm(config):
    import tensorflow as tf

    input = Input(shape=(sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embedding_matrix])(input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    output = Dense(3, activation="softmax")(conc)

    # this creates a model that includes
    model = Model(inputs=input, outputs=output)
    adam = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def init_estimator(model="text_cnn"):
    if model == "text_cnn":
        est = Estimator.from_keras(
            model_creator=model_creator_text_cnn, workers_per_node=1)
    elif model == "gru":
        est = Estimator.from_keras(
            model_creator=model_creator_gru, workers_per_node=1)
    elif model == "lstm":
        est = Estimator.from_keras(
            model_creator=model_creator_lstm, workers_per_node=1)
    return est


def stop_context():
    stop_orca_context()


def convert_prediction(predictions):
    to_array = f.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    predictions = predictions.withColumn('prediction', to_array('prediction'))
    predictions = predictions.select(
        'timestamp',
        'user',
        'comment',
        f.expr('array_position(cast(prediction as array<float>), cast(array_max(prediction) as float)) - 1').alias("prediction")
    )
    return predictions
