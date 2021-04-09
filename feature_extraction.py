from keras.applications.resnet50 import ResNet50, preprocess_input

def resnet_feature_extractor():
    video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    for layer in model.layers:
        layer.trainable = False

    encoded_frame_sequence = TimeDistributed(model)(video_input)

    feature_extract_model = Model(inputs=video_input, outputs=encoded_frame_sequence)

    return feature_extract_model

