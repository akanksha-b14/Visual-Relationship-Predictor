import json
import os
import numpy as np

from PIL import Image
from numpy import asarray


def read_videos(video_ids, base_dir):
    frames = []
    image_files = []
    for video_id in video_ids:
        for img_file in os.listdir(base_dir + "/" + video_id):
            image_files.append(img_file)
        image_files.sort()
        frame_data = []
        for i in range(0, 30):
            img_file = image_files[i]
            img_path = base_dir + "/" + video_id + "/" + img_file
            img = Image.open(img_path)
            numpydata = asarray(img)
            frame_data.append(numpydata)

        frames.append(np.array(frame_data))

    return np.array(frames)


def get_object_idx_map():
    with open('object1_object2.json') as f:
        object_idx_map = json.load(f)
    idx_object_map = {}
    for key in object_idx_map.keys():
        idx_object_map[object_idx_map[key]] = key

    return object_idx_map, idx_object_map


def get_relationship_idx_map():
    with open('relationship.json') as f:
        relationship_idx_map = json.load(f)
    idx_relation_map = {}
    for key in relationship_idx_map.keys():
        idx_relation_map[relationship_idx_map[key]] = key

    return relationship_idx_map, idx_relation_map


def get_given_annotations():
    with open('training_annotation.json') as f:
        given_annotation = json.load(f)
    return given_annotation


def get_video_ids(base_dir):
    video_ids = []
    for ids in os.listdir(base_dir):
        video_ids.append(ids)
    return video_ids


object_idx_map, idx_object_map = get_object_idx_map()
relationship_idx_map, idx_relation_map = get_relationship_idx_map()

vocab = list(object_idx_map.keys()) + list(relationship_idx_map.keys())
vocab_idx_map = {}
idx = 0
for word in vocab:
    vocab_idx_map[word] = idx
    idx += 1

def get_idx_vocab_map():
    idx_vocab_map = {}
    for key in vocab_idx_map.keys():
        idx_vocab_map[vocab_idx_map[key]] = key
    return idx_vocab_map

def get_relationship_idx():
    relationship_idx = []
    for idx in idx_relation_map.keys():
        relationship_idx.append(vocab_idx_map[idx_relation_map[idx]])

    return relationship_idx

def get_object_idx():
    object_idx = []
    for idx in idx_object_map.keys():
        object_idx.append(vocab_idx_map[idx_object_map[idx]])

    return object_idx


def get_new_annotation():
    given_annotation = get_given_annotations()
    object_idx_map, idx_object_map = get_object_idx_map()
    relationship_idx_map, idx_relation_map = get_relationship_idx_map()

    new_annotation = {}
    for key in given_annotation.keys():
        value = given_annotation[key]
        new_annotation[key] = [vocab_idx_map[idx_object_map[value[0]]], vocab_idx_map[idx_relation_map[value[1]]],
                               vocab_idx_map[idx_object_map[value[2]]]]

    return new_annotation

def validation_accuracy(y_pred, relationship):
    valid_pred = np.zeros_like(y_pred)
    relationship_idx = get_relationship_idx()
    object_idx = get_object_idx()

    if relationship == 1:
        for idx in relationship_idx:
            valid_pred[idx] = y_pred[idx]
    else:
        for idx in object_idx:
            valid_pred[idx] = y_pred[idx]
    pred_idx = valid_pred.argsort()[-5:][::-1]

    return pred_idx
