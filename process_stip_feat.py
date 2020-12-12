import numpy as np
import cv2
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import Normalizer        
	
#### Enable this, in order to print the full size of the ndarrays created. ####
#np.set_printoptions(threshold='nan')

#### Class names you want to extract features from. ####
videos = ['ApplyEyeMakeup', 'SoccerPenalty', 'BabyCrawling', 'Typing', 'BlowingCandles', 'Haircut', 'FloorGymnastics',
		  'BrushingTeeth', 'FieldHockeyPenalty', 'HeadMassage', 'ParallelBars', 'HandstandWalking']

#### Create the file list from the directory where the dataset's STIP text files are, and make it your working directory. ####
file_list = os.listdir(r'/<your_file_directory>')
os.chdir(r'/<your_file_directory>')

#### Define the dictionary size for the Bag of Words framework. In this project we use 4000. ####
dictionarySize_stip = 4000

#### List containing all the extracted stip features. ####
stips = []
#### List containing the extracted stip features to be used to create the dictionary. ####
stipst = []
#### Number of video clips processed. ####
n = 0

for video in videos:
#### Open the text file for each category and read the whole matrix. #### 
	fo = open('%s.txt' % (video), 'r')
	stip_feat = np.loadtxt(fo)
	fo.close()

#### Counting features for each video clip. ####
	count = 0
#### Number of features for all video clips in the text file. ####
	n_count = 0
#### List of numbers of features for each video clip. ####
	points = []

	with open("%s.txt" % (video)) as fileobject:
		for line in fileobject:
			if not line.startswith("#") and not line.isspace() and line:
				count += 1
				n_count += 1
			elif (count != 0):
				print ("%d : %d " % (len(points)+1, count))
				points.append(count)
				count = 0
#### Append the final clip's number of features. ####
	points.append(count)

#### Index to browse the points list. ####
	index = 0

#### Process all video clips from each class. ####
	for i in range(1,26):
		for j in range(1,8):
			name = 'v_%s_g%02d_c%02d' % (video, i, j)
			if ('%s.avi' % (name)) not in file_list:
				break
			else:
				n += 1
				print ("%d : %s" % (n, name))
#### Keep the stip features from this clip. ####
				my_stip, stip_feat = np.vsplit(stip_feat, [points[index]])
#### Keep only the HOG, HOF descriptors for each feature. ####
				rest_stip, my_stip = np.hsplit(my_stip, [9])
				my_stip = np.float32(my_stip)
				stips.append(my_stip)
#### We choose the clips of the 16 first videos of each class, to create the dictionary. #### 
				if (i<16):
					stipst.append(my_stip)
				index += 1

#### We perform PCA analysis to the videos chosen to create the dictionary. ####
stip_ar = np.vstack(stipst)
stip_ar = np.float32(stip_ar)
from sklearn.decomposition import PCA
#### In this project, we chose 99% acceptable variance for the PCA. ####
pca = PCA(n_components=0.99, whiten=True, svd_solver='auto')
pcaed = pca.fit_transform(stip_ar)
print ("PCA components are: %d" % (pca.n_components_))

#### We perform K-means clustering to produce the dictionary. ####
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness1,labels1,centers = cv2.kmeans(pcaed,dictionarySize_stip,None,criteria,10,flags)
dictionary_stip = centers

#### We encode all the extracted features using Vector Quantization, using the dictionary. ####
all_stip = np.zeros((n, dictionarySize_stip), 'float32')
for i in range(n):
	words, distance = vq(pca.transform(stips[i]), dictionary_stip)
	for w in words:
		all_stip[i][w] += 1

#### We L2-normalize the final matrix ####
all_stip = Normalizer().transform(all_stip)

#### Saving in a text file. ####
fo = open('Final_stip_bag.txt', 'wb')
np.savetxt(fo, all_stip)
fo.close()
