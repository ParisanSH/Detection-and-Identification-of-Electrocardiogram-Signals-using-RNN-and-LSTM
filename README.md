# Detection and Identification of Electrocardiogram Signals using Recurrent Neural Networks(RNN) & LSTM
In this project, I tried to predict the secondary sequences (annotations) from their primary sequences (samples). A dataset of 146 files is uploaded. Each sample file contains a primary sequence, ‘mV’ of signal, and its secondary is in the annotation file ‘annotation’. 
Each primary sequence contains data streams of continuous values and its secondary sequence consists of 0 to 3 (1 for the P wave, 2 for the QRS complex, 3 for the T wave, and 0 for other points of signal). Thus, there are 73 input and output patterns of data streams. 

# Electrocardiogram
An electrocardiogram records the electrical activity of the heart at each contraction. When an electric wave is generated in the heart, the inside of the heart cell quickly becomes positive relative to the outside. Stimulation by an electric wave nourishes the polarity of the cell.
The most important characteristic of ECG signal that cardiologists employ in the diagnosis of heart disease is the QRS complex. This characteristic is more important than T and P waves and other signal properties because it is easier to distinguish and separate from the ECG signal than other features. Also, the QRS complex shows ventricular depolarization, which plays the most important role in the electrical activity of the heart. Therefore, the diagnosis and isolation of the compound are crucial in the classification and diagnosis of cardiac abnormalities. Also, by diagnosing and counting QRS, the intensity of the heartbeat, and its possible inconsistencies can be observed and examined.

# Steps for RNN:
1. The first 80% of each data sequence file (primary and secondary) is used for training and the rest 20% for testing.
2. A sliding window of size 11 is used for each sequence to obtain the training samples.
3. An Elman neural network is designed, trained, and tested.
4. A NARX neural network is designed, trained, and tested.
5. The training and test accuracy of each method is reported.
6. Sliding windows of sizes 5 and 21 are used, and steps 3 to 5 are repeated.

# Steps for LSTM:
1. First, each file (with 15000 or 2500 samples) has been divided into records with 500 samples. ( the samples should not be shuffled!)
2. Then, there are 100 sample records with annotation. A sliding window of size 5 has been used in order to get training patterns for each record.
3. After the training patterns is prepared, a bidirectional LSTM network has been built for classification and training. (Training sequences of each record forward and backward have been presented to two separate LSTM layers, both of which are connected to the same output layer)
4. steps 1 and 2 have been repeated for the test data and then they became classified.
5. The classification accuracy of the predictions has been calculated.

