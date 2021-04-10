from keras.layers import Input, Concatenate
from keras.layers import LSTM, Bidirectional
from keras.layers.core import Dense
from keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from feature_extraction import resnet_feature_extractor

class EncoderDecoder:
    def __init__(self, output_seq_length, feature_vector_length):
        self.output_seq_length = output_seq_length
        self.feature_vector_length = feature_vector_length
        self.encoder = Bidirectional(LSTM(100, return_state=True, dropout=0.5, recurrent_dropout=0.5))
        self.decoder_input_dense = Dense(200, activation = 'relu')
        self.decoder = LSTM(200, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
        self.decoder_output_dense =  Dense(output_seq_length, activation='softmax')
        self.feature_extractor = resnet_feature_extractor()

    def load_training_model(self):
        # encoder_inputs = Input(shape=(None, self.feature_vector_length))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(self.feature_extractor.outputs)

        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.output_seq_length))
        decoder_input_feature = self.decoder_input_dense(decoder_inputs)
        decoder_outputs, _, _ = self.decoder(decoder_input_feature, initial_state=encoder_states)
        decoder_outputs = self.decoder_output_dense(decoder_outputs)
        model = Model([self.feature_extractor.inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def load_test_encoder(self):
        # encoder_inputs = Input(shape=(None, self.feature_vector_length))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(self.feature_extractor.outputs)

        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        encoder_model = Model(self.feature_extractor.inputs, encoder_states)

        return encoder_model

    def load_test_decoder(self):
        decoder_state_input_h = Input(shape=(200,))
        decoder_state_input_c = Input(shape=(200,))
        decoder_inputs = Input(shape=(None, self.output_seq_length))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_input_feature = self.decoder_input_dense(decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder(decoder_input_feature, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_output_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return decoder_model
