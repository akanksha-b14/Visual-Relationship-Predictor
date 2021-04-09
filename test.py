import matplotlib.pyplot as plt
test_base_dir = '/kaggle/input/cs-5242-project-nus-2021-semester2/test/test'

def validation_accuracy(y_pred, relationship):
    valid_pred = np.zeros_like(y_pred)
    if relationship == 1:
        for idx in relationship_idx:
            valid_pred[idx] = y_pred[idx]
    else:
        for idx in object_idx:
            valid_pred[idx] = y_pred[idx]
    pred_idx = valid_pred.argsort()[-5:][::-1]
    return pred_idx

def decode_sequence(input_seq, result, count):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    for i in range(0,3):
        decoder_input = [target_seq] + states_value
        output_tokens, h, c = decoder_model.predict(decoder_input)

        result[i][count] = output_tokens[0][0]
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]



test_video_ids = get_video_ids(test_base_dir)
test_video_ids.sort()
result = []
count = 0
for id in test_video_ids:
    print(id)
    video = read_videos([id],test_base_dir)
    pred = np.zeros((3,3,117))
    pred_string = ""
    for i in range(0,3):
        if i == 0:
            aug_video = video[0]
        else:
            aug_video = augment(video[0])
        plt.imshow(aug_video[15])
        plt.show()
        input_seq = feature_model.predict([preprocess_videos(aug_video)])
        decode_sequence(input_seq, pred, i)
    for i in range(0,3):
        row = {}
        row['ID'] = count + i
        ids = []
        predictions = validation_accuracy(np.max(pred[i],axis=0), i)
        for id in predictions:
            if i == 1:
                ids.append(str(relationship_idx_map[idx_vocab_map[id]]))
            else:
                ids.append(str(object_idx_map[idx_vocab_map[id]]))
        row['label'] = " ".join(ids)
        result.append(row)
        pred_string = pred_string + " " + str(idx_vocab_map[predictions[0]]) + "/" + str(idx_vocab_map[predictions[1]])

        count = count + 1
        
    print(pred_string)