import random
import numpy as np
import feature_extraction
import utils
import pandas as pd

from model import EncoderDecoder
from data_augmentation import augment

train_base_dir = '../train/train'
max_encoder_seq_length = 30
max_decoder_seq_length = 3
feature_vector_length = 2048
output_seq_length = 117
NUM_FRAMES = 30

video_ids = utils.get_video_ids(train_base_dir)
new_annotation = utils.get_new_annotation()
encoder_decoder = EncoderDecoder(output_seq_length, feature_vector_length)
input_videos = []
target_texts = []

# training model on the given train data
for id in video_ids:
    input_video = id
    target_text = new_annotation[id]
    input_videos.append(input_video)
    target_texts.append(target_text)
    
encoder_input_data = np.zeros((len(input_videos), max_encoder_seq_length, feature_vector_length), dtype='float32')
decoder_input_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')

feature_model = feature_extraction.resnet_feature_extractor()
for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
    video = feature_extraction.load_videos([input_video],train_base_dir)
    encoder_input_data[i] = feature_model.predict([video])
    for t, annotation in enumerate(target_text):
        decoder_target_data[i, t, annotation] = 1
        if t > 0:
            decoder_input_data[i, t] = decoder_target_data[i, t-1]

batch_size = 5
epochs = 8
model = encoder_decoder.load_training_model()
print(model.summary())
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
         batch_size=batch_size,
         epochs=epochs)


# training model on the augmented data
batch_size = 5
epochs = 2
random.shuffle(video_ids)
for j in range(0, 10):
    input_videos = []
    target_texts = []
    
    for id in video_ids:
        input_videos.append(id)
        target_texts.append(new_annotation[id])
    
    encoder_input_data = np.zeros((len(input_videos), max_encoder_seq_length, feature_vector_length), dtype='float32')
    decoder_input_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(input_videos), max_decoder_seq_length, output_seq_length), dtype='float32')

    for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
        video = utils.read_videos([input_video],train_base_dir)
        aug_video = augment(video[0])
        encoder_input_data[i] = feature_model.predict([feature_extraction.preprocess_videos(aug_video)])
        for t, annotation in enumerate(target_text):
            decoder_target_data[i, t, annotation] = 1
            if t > 0:
                decoder_input_data[i, t] = decoder_target_data[i, t-1]


    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
             batch_size=batch_size,
             epochs=epochs, validation_split=0.2)
    
    print("*************** Next Augmentation **********************")

#predicting results on test data
test_base_dir = '../test/test'
relationship_idx_map = utils.get_relationship_idx_map()[0]
object_idx_map = utils.get_object_idx_map()[0]
idx_vocab_map = utils.get_idx_vocab_map()
encoder_model = encoder_decoder.load_test_encoder()
decoder_model = encoder_decoder.load_test_decoder()

def decode_sequence(input_seq, result, count):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, output_seq_length))
    for i in range(0, 3):
        decoder_input = [target_seq] + states_value
        output_tokens, h, c = decoder_model.predict(decoder_input)

        result[i][count] = output_tokens[0][0]
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        target_seq = np.zeros((1, 1, output_seq_length))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]


test_video_ids = utils.get_video_ids(test_base_dir)
test_video_ids.sort()
result = []
count = 0
for vid_id in test_video_ids:
    video = utils.read_videos([vid_id], test_base_dir)
    pred = np.zeros((3, 3, 117))
    for i in range(0, 3):
        if i == 0:
            aug_video = video[0]
        else:
            aug_video = augment(video[0])
        input_seq = feature_model.predict([feature_extraction.preprocess_videos(aug_video)])
        decode_sequence(input_seq, pred, i)
    for i in range(0, 3):
        row = {}
        row['ID'] = count + i
        ids = []
        predictions = utils.validation_accuracy(np.max(pred[i], axis=0), i)
        for id in predictions:
            if i == 1:
                ids.append(str(relationship_idx_map[idx_vocab_map[id]]))
            else:
                ids.append(str(object_idx_map[idx_vocab_map[id]]))
        row['label'] = " ".join(ids)
        result.append(row)

        count = count + 1


df = pd.DataFrame(result)
print(df)
df.to_csv('predictions.csv', index=False)
