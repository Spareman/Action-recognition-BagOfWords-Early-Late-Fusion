# Action-recognition-BagOfWords-Early-Late-Fusion
Video classification using the UCF101 dataset for action recognition. We extract SIFT, MFCC and STIP features from the videos, we encode them using the Bag of Words framework and we implement early and late feature fusion using different combinations of the feature types available.

The code is part of my Bachelor's thesis submited at the School of Electrical and Computer Engineering of the National Technological University of Athens, with the title "Video Action Recognition using Bag of Words and feature fusion".

# Abstract
Nowadays, Artificial Intelligence enters our everyday lives in a rapid pace and the field of Computer Vision has experienced great growth, while research constantly improves the way that computers understand and analyze the visual clues which they receive. Multimedia Action Recognition has received attention of the research community. Its aim is to develop a system that detects human actions that appear in a video, picture etc. The term “action” means a basic person-related interaction with meaning and it might include the simplest actions, like “Walking”, or maybe more complex, like “Playing Soccer”.

In this thesis, we develop an action recognition system, which extracts visual, sound and motion features for video representation and uses the well-known Bag of Words framework to represent these features using a codebook consisting of fragments of train data. This codebook is used to encode the train data, creating a robust representation with a single vector for each video. This technique benefits the training process of a classifier, which in our case is a Support Vector Machine. The classifier predicts the action classes in which each video belongs.

Moving on, we have experimented different feature fusion methods in order to achieve a more representative representation and finally to improve the average accuracy of our system. Specifically, we have implemented early fusion methods as well as late fusion methods, with or without a meta-classifier. Furthermore, we checked the combination of different fusion categories.

Our results highlight the significance of a proper preprocessing phase of our data before training the classifiers in order to achieve an acceptable level of generalization. Moreover, we conclude that even the simplest implementation of fusion of complementary features can result an important improvement in the average accuracy of our system. Our experimental results encourage further research towards this direction.
