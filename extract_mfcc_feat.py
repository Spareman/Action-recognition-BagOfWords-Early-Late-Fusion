import numpy as np
import cv2
import os
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
from sklearn.preprocessing import Normalizer
from scipy.cluster.vq import vq, kmeans, whiten

	
#### Enable this, in order to print the full size of the ndarrays created. ####
#np.set_printoptions(threshold='nan')


#### Class names you want to extract features from. ####
videos = ['ApplyEyeMakeup', 'SoccerPenalty', 'BabyCrawling', 'Typing', 'BlowingCandles', 'Haircut',
			'FloorGymnastics', 'BrushingTeeth', 'FieldHockeyPenalty', 'HeadMassage', 'ParallelBars', 'HandstandWalking']

#### Create the file list from the directory where the dataset's videos are, and make it your working directory. ####
file_list = os.listdir(r'/<your_file_directory>/UCF101/')
os.chdir(r'/<your_file_directory>/UCF101/')

#### Define the dictionary size for the Bag of Words framework. In this project we use 4000. ####
dictionarySize_mfcc = 4000

#### List containing all the extracted mfcc features. ####
mfccs = []
#### List containing the extracted mfcc features to be used to create the dictionary. ####
mfccst = []
#### Number of video clips processed. ####
n = 0

#### Process all video clips from the classes in the videos list. ####
for video in videos:
	for i in range(1,26):
		for j in range(1,8):
			name = 'v_%s_g%02d_c%02d' % (video, i, j)
			if ('%s.avi' % (name)) not in file_list:
				break
			else:
				print (name)
				n += 1
				(rate, sig) = wav.read('%s.wav' % (name))
#### In this project we use 32msec window length and 16msec step. ####
				mfcc_feat = mfcc(sig, rate, 0.032, 0.016, nfft=1500)
#### We compute the deltas and delta deltas of the extracted coefficients. ####
				deltas = delta(mfcc_feat, 2)
				ddeltas = delta(deltas, 2)
#### We concatenate all the above in a single vector. ####
				mfccs.append(np.hstack([mfcc_feat, deltas, ddeltas]))
#### We choose the clips of the 16 first videos of each class, to create the dictionary. #### 
				if (i<16):
					mfccst.append(np.hstack([mfcc_feat, deltas, ddeltas]))

#### We perform PCA analysis to the videos chosen to create the dictionary. ####
mfcc_ar = np.float32(np.vstack(mfccst))
from sklearn.decomposition import PCA
#### In this project, we chose 99% acceptable variance for the PCA. ####
pca = PCA(n_components=0.99, whiten=True, svd_solver='auto')
pcaed = pca.fit_transform(mfcc_ar)
print ("PCA components are: %d" % (pca.n_components_))

#### We perform K-means clustering to produce the dictionary. ####
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness1,labels1,centers = cv2.kmeans(pcaed,dictionarySize_mfcc,None,criteria,10,flags)
dictionary_mfcc = centers

#### We encode all the extracted features using Vector Quantization, using the dictionary. ####
all_mfcc = np.zeros((n, dictionarySize_mfcc), 'float32')
for i in range(n):
	words, distance = vq(pca.transform(mfccs[i]), dictionary_mfcc)
	for w in words:
		all_mfcc [i][w] += 1

#### We L2-normalize the final matrix ####
all_mfcc = Normalizer().transform(all_mfcc)

#### Saving in a text file. ####
fo = open('Final_mfcc_bag.txt', 'wb')
np.savetxt(fo, all_mfcc)
fo.close()