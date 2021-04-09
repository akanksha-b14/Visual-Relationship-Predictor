from keras.layers import LSTM
from keras.layers import TimeDistributed, Input, concatenate, Flatten, Embedding
from keras.layers.core import Dense
from keras.models import Model, Sequential
from keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def lstm_model():    
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # encoder_input_feature = Dense(2058, activation = 'relu')(encoder_inputs)
    encoder = Bidirectional(LSTM(100, return_state=True, dropout=0.5, recurrent_dropout=0.5))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_input_dense = Dense(200, activation = 'relu')
    decoder_input_feature = decoder_input_dense(decoder_inputs)
    decoder_lstm = LSTM(200, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_input_feature, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])