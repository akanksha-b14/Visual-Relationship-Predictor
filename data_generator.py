import numpy as np
from feature_extraction import load_videos

max_encoder_seq_length = 30
max_decoder_seq_length = 3
feature_vector_length = 2048
output_seq_length = 117

train_base_dir = '../train/train'

def read_data(input_videos, target_texts):
    encoder_input_data = np.zeros((len(input_videos), max_encoder_seq_length, 224, 224, 3), dtype='float32')
    decoder_input_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')


    for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
        video = load_videos([input_video],train_base_dir)
        encoder_input_data[i] = video
        for t, annotation in enumerate(target_text):
            decoder_target_data[i, t, annotation] = 1
            if t > 0:
                decoder_input_data[i, t] = decoder_target_data[i, t-1]

    return [encoder_input_data,decoder_input_data], decoder_target_data

from keras.utils import Sequence

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x,y = read_data(batch_x, batch_y)
        return x, y