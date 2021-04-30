from global_vars import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

class BidirectionalModel:
    def __init__(self):
        pass

    def create_model(self):
        pass

    def train_model(self, data):
        pass

    def create_batch(self, input_data):
        pass

model = Sequential()

# defining layer 1 forward LSTM layer
lstm_fw_layer1 = LSTM(
    CELLS, 
    dropout=DROPOUT_PROBABILITY, 
    activation='relu', 
    return_sequences=True)
# defining full BLSTM layer
blstm_layer1 = Bidirectional(lstm_fw_layer1)

# defining layer 2 forward LSTM layer
lstm_fw_layer2 = LSTM(
    CELLS, 
    dropout=DROPOUT_PROBABILITY, 
    activation='relu', 
    return_sequences=True)
# defining full BLSTM layer
blstm_layer2 = Bidirectional(lstm_fw_layer2)

# dense layer
feedforward_layer = Dense(EMBEDDING_DIM, activation='tanh')

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