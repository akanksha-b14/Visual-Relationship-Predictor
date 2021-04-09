import os

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import preprocess_input
from keras.layers import TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model



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

def preprocess_videos(read_frames):
    frame_data = []
    for i in range(0,30):
        img = read_frames[i]
        img = image.smart_resize(img, (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        frame_data.append(x.reshape(224, 224, 3))

    return np.expand_dims(np.array(frame_data),axis=0)

def resnet_feature_extractor():
    video_input = Input(shape=(30, 224, 224, 3))
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    for layer in model.layers:
        layer.trainable = False

    encoded_frame_sequence = TimeDistributed(model)(video_input)

    feature_extract_model = Model(inputs=video_input, outputs=encoded_frame_sequence)

    return feature_extract_model

