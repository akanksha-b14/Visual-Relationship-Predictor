import os

import numpy as np

train_base_dir = '/kaggle/input/cs-5242-project-nus-2021-semester2/train/train'
input_videos = []
target_texts = []

def load_videos(video_ids, base_dir):
    frames = []
    image_files = []
    for video_id in video_ids:
        for img_file in os.listdir(base_dir + "/" + video_id):
            image_files.append(img_file)
        image_files.sort()
        frame_data = []
        for i in range(0,30):
            img_file = image_files[i]
            img_path = base_dir + "/" + video_id + "/" + img_file
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            frame_data.append(x.reshape(224, 224, 3))

        frames.append(np.array(frame_data))
    return np.array(frames)

video_ids = get_video_ids(train_base_dir)
num_samples = len(video_ids)

new_annotation = get_new_annotation()

object_idx_map, idx_object_map = get_object_idx_map()
relationship_idx_map, idx_relation_map = get_relationship_idx_map()

vocab = list(object_idx_map.keys()) + list(relationship_idx_map.keys())
vocab_idx_map = {}
idx = 0
for word in vocab:
    vocab_idx_map[word] = idx
    idx += 1
idx_vocab_map = {}
for key in vocab_idx_map.keys():
    idx_vocab_map[vocab_idx_map[key]] = key


relationship_idx = []
for idx in idx_relation_map.keys():
    relationship_idx.append(vocab_idx_map[idx_relation_map[idx]])

object_idx = []
for idx in idx_object_map.keys():
    object_idx.append(vocab_idx_map[idx_object_map[idx]])


max_encoder_seq_length = 30
max_decoder_seq_length = 3
num_encoder_tokens = 2048
num_decoder_tokens = 117
NUM_FRAMES = 30

np.random.seed(5242)


feature_model = resnet_feature_extractor()
feature_model.summary()
