from global_vars import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

from tensorflow.keras.utils import plot_model

class BidirectionalModel:
    def __init__(self, n_hidden, batch_size, fw_keep_prob, rec_keep_prob):
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.fw_keep_prob = fw_keep_prob
        self.rec_keep_prob = rec_keep_prob

    def create_model(self):
        print(CELLS)
        print(DROPOUT_PROBABILITY)
        print(EMBEDDING_DIM)
        print(LEARNING_RATE)
        print(DECAY)
        model = Sequential()
        model._name="Practice_Model"

        # defining layer 1 forward LSTM layer
        lstm_fw_layer1 = LSTM(
            CELLS, 
            dropout=DROPOUT_PROBABILITY, 
            activation='relu', 
            return_sequences=True)
        # defining full BLSTM layer
        blstm_layer1 = Bidirectional(lstm_fw_layer1, input_shape=(self.batch_size, 1), name="LSTM_Layer_1")

        # defining layer 2 forward LSTM layer
        lstm_fw_layer2 = LSTM(
            CELLS, 
            dropout=DROPOUT_PROBABILITY, 
            activation='relu', 
            return_sequences=True)
        # defining full BLSTM layer
        blstm_layer2 = Bidirectional(lstm_fw_layer2, input_shape=(self.batch_size, 1), name="LSTM_Layer_2")

        # dense layer
        feedforward_layer = Dense(EMBEDDING_DIM, activation='tanh', name="Embedding_Layer")

        # creating the model
        model.add(blstm_layer1)
        model.add(blstm_layer2)
        model.add(feedforward_layer)

        opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=DECAY)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        print('Model compiled successfully')        
        
        return model

    def train_model(self, data):
        pass

    def create_batch(self, input_data):
        pass

if __name__ == '__main__':
    bi_model = BidirectionalModel(3, 300, 0.1, 0.1)
    model = bi_model.create_model()
    model.build()
    print(model.summary())

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)