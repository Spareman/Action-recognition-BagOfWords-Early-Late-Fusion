import numpy as np
import cv2
import os
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.preprocessing import Normalizer
from skimage.util.shape import view_as_blocks

#### Enable this, in order to print the full size of the ndarrays created. ####
#np.set_printoptions(threshold='nan')

#### Class names you want to extract features from. ####
videos = ['ApplyEyeMakeup', 'SoccerPenalty', 'Typing', 'Haircut', 'FloorGymnastics','BrushingTeeth',
		  'FieldHockeyPenalty', 'HeadMassage', 'ParallelBars', 'HandstandWalking', 'BabyCrawling', 'BlowingCandles']

#### Create the file list from the directory where the dataset's videos are, and make it your working directory. ####
file_list = os.listdir(r'/<your_file_directory>/UCF101/')
os.chdir(r'/<your_file_directory>/UCF101/')

#### Define the dictionary size for the Bag of Words framework. In this project we use 500. ####
dictionarySize_sift = 500

#### List containing all the extracted sift features. ####
sifts = [[] for k in range(5)]
#### List containing the extracted stip features to be used to create the dictionary. ####
siftst = [[] for k in range(5)]
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

#### Read the video clip. ####
				cap = cv2.VideoCapture("%s.avi" % (name))
				fps = cap.get(cv2.CAP_PROP_FPS)
				n += 1
#### Counting number of frames to sample, while sampling the first one. ####
				count = fps-1
#### Temporary list of extracted features of each frame. ####
				img_temp = [[] for k in range(5)]
				while (cap.isOpened()):

					(ret, frame) = cap.read()
					if (ret):
						count += 1
					else:
						break
					if (cv2.waitKey(1) & 0xFF == ord('q')):
						break
					if (count == fps):
						count = 0
						img = frame
						gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
						pyramid = view_as_blocks(gray, block_shape=(gray.shape[0]/2, gray.shape[1]/2))
						sift = cv2.xfeatures2d.SIFT_create()
#### Creating a 2x2 spatial pyramid of the frame. ####
						img_list = [gray, pyramid[0][0], pyramid[0][1], pyramid[1][0], pyramid[1][1]]
						for im in range(5):
							(kp, des) = sift.detectAndCompute(img_list[im], None)
							if (des is not None):
								img_temp[im].append(des)

						#img = cv2.drawKeypoints(gray, kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
						#BOW_sift.add(des)
				cap.release()
#### Creating a list of the extracted features for each region for all video clips. ####
				for im in range(5):
					if (img_temp[im] != []):
						temp = np.vstack(img_temp[im])
					else:
						#temp = np.zeros((1,128))
						temp = np.full((1,128), np.nan)
					sifts[im].append(temp)
#### We choose the clips of the 16 first videos of each class, to create the dictionary. #### 
					if (i<16):
						siftst[im].append(temp)
cv2.destroyAllWindows()

from sklearn.decomposition import PCA
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="most_frequent")

for i in range(len(sifts)):
#### We perform PCA analysis to the videos chosen to create the dictionary. ####
	sift_ar = np.float32(np.vstack(siftst[i]))
	sift_ar = imp.fit_transform(sift_ar)
	pca = PCA(n_components=0.99, whiten=True, svd_solver='auto')
	pcaed = pca.fit_transform(sift_ar)
	print ("PCA components are: %d" % (pca.n_components_))

#### We perform K-means clustering to produce the dictionary. ####
	compactness, labels, centers = cv2.kmeans(pcaed, dictionarySize_sift, None, criteria, 10, flags)
	dictionary_sift = centers

#### We encode all the extracted features using Vector Quantization, using the dictionary. ####
	temp_sift = np.zeros((n, dictionarySize_sift), 'float32')
	for j in range(n):
		words, distance = vq(pca.transform(imp.transform(sifts[i][j])), dictionary_sift)
		for w in words:
			temp_sift[j][w] += 1

#### We L2-normalize the final matrix ####
	temp_sift = Normalizer().transform(temp_sift)

	if (i == 0):
		all_sift = temp_sift
	else:
#### We concatenate the produced vectors from each region in a single vector for each video clip. ####
		all_sift = np.hstack([all_sift, temp_sift])

#### Saving in a text file. ####
fo = open('Final_sift_bag.txt', 'wb')
np.savetxt(fo, all_sift)
fo.close()