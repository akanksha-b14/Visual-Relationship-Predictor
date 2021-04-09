import json
import albumentations as A

def get_object_idx_map():
    with open('/kaggle/input/cs-5242-project-nus-2021-semester2/object1_object2.json') as f:
        object_idx_map = json.load(f)
    idx_object_map = {}
    for key in object_idx_map.keys():
        idx_object_map[object_idx_map[key]] = key

    return object_idx_map, idx_object_map

def get_relationship_idx_map():
    with open('/kaggle/input/cs-5242-project-nus-2021-semester2/relationship.json') as f:
        relationship_idx_map = json.load(f)
    idx_relation_map = {}
    for key in relationship_idx_map.keys():
        idx_relation_map[relationship_idx_map[key]] = key

    return relationship_idx_map, idx_relation_map

def get_given_annotations():
    with open('/kaggle/input/cs-5242-project-nus-2021-semester2/training_annotation.json') as f:
        given_annotation = json.load(f)
    return given_annotation


def get_new_annotation():
    given_annotation = get_given_annotations()
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
    new_annotation = {}
    for key in given_annotation.keys():
        value = given_annotation[key]
        new_annotation[key] = [vocab_idx_map[idx_object_map[value[0]]],vocab_idx_map[idx_relation_map[value[1]]],vocab_idx_map[idx_object_map[value[2]]]]

    return new_annotation

def get_video_ids(base_dir):
    video_ids = []
    for ids in os.listdir(base_dir):
        video_ids.append(ids)
    return video_ids

transform = A.ReplayCompose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=(-15,15), p=0.5),
    A.Downscale(p=0.2),
    A.RandomResizedCrop(224, 224, scale=(0.7,1.2))
])

##For data augmentation
def augment(frames):
    aug_frames = []                 
    for i in range(0,30):
        if i == 0:
            data = transform(image=frames[i])
            img = data['image']
        else:
            replay_image = A.ReplayCompose.replay(data['replay'], image=frames[i])
            img = replay_image['image']
            
        aug_frames.append(img)
        
    return np.array(aug_frames)