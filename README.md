# Visual Relationship Predictor
This project was done as part of coursework at National University of Singapore.

The aim of the project was to predict relationships between different objects present in frames extracted from a video. 

For the purpose of this task we used a sequence-to-sequence model approach. The input was a sequence of frames and the output is relation triplet of the form: <object1-relationship-object2>.
We used ResNet152 to get features from frames and the same were passed to a bi-directional LSTM encoder-decoder model.

The best accuracy reported for this task was 71.63%
