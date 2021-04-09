from model import lstm_model
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

batch_size =  5
epochs = 8

for id in video_ids:
    input_video = id
    target_text = new_annotation[id]
    input_videos.append(input_video)
    target_texts.append(target_text)
    
encoder_input_data = np.zeros((len(input_videos), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_videos), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_videos), max_decoder_seq_length, num_decoder_tokens), dtype='float32')


for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
    video = load_videos([input_video],train_base_dir)
    encoder_input_data[i] = feature_model.predict([video])
    for t, annotation in enumerate(target_text):
        decoder_target_data[i, t, annotation] = 1
        if t > 0:
            decoder_input_data[i, t] = decoder_target_data[i, t-1]
            
model = lstm_model()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
         batch_size=batch_size,
         epochs=epochs)

def read_videos(video_ids, base_dir):
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
            img = Image.open(img_path)
            numpydata = asarray(img)
            frame_data.append(numpydata)
        
        frames.append(np.array(frame_data))
    
    return np.array(frames)
    

def preprocess_videos(read_frames):
    frame_data = []
    for i in range(0,30):
        img = read_frames[i]
#         if i == 0:
#             plt.imshow(img)
#             plt.show()
#         img = image.load_img(img_path, target_size=(224, 224))
        img = image.smart_resize(img, (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        frame_data.append(x.reshape(224, 224, 3))

    return np.expand_dims(np.array(frame_data),axis=0)
import random
batch_size =  5
epochs = 2

random.shuffle(video_ids)
for j in range(0,10):

    input_videos = []
    target_texts = []
    
    for id in video_ids:
        input_videos.append(id)
        target_texts.append(new_annotation[id])
    
    encoder_input_data = np.zeros((len(input_videos), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_videos), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_videos), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
        video = read_videos([input_video],train_base_dir)
        aug_video = augment(video[0])
        encoder_input_data[i] = feature_model.predict([preprocess_videos(aug_video)])
        for t, annotation in enumerate(target_text):
            decoder_target_data[i, t, annotation] = 1
            if t > 0:
                decoder_input_data[i, t] = decoder_target_data[i, t-1]


    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
             batch_size=batch_size,
             epochs=epochs, validation_split=0.2)
    
    print("*************** Next Augmentation **********************")