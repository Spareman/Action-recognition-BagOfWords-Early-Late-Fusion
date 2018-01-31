import numpy as np
import cv2
import os
import pickle

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import chi2_kernel
class Chi2Kernel(BaseEstimator, TransformerMixin):
	def __init__(self, gamma=1.0):
		super(Chi2Kernel, self).__init__()
		self.gamma = gamma

	def transform(self, X):
		return chi2_kernel(X, self.X_train_, gamma=self.gamma)

	def fit(self, X, y=None, **fit_params):
		self.X_train_ = X
		return self

#### Enable this, in order to print the full size of the ndarrays created. ####
#np.set_printoptions(threshold='nan')

print ('Lets start')

#### Class names you want to extract features from. ####
videos = ['ApplyEyeMakeup', 'SoccerPenalty', 'Typing', 'Haircut', 'FloorGymnastics','BrushingTeeth',
		  'FieldHockeyPenalty', 'HeadMassage', 'ParallelBars', 'HandstandWalking', 'BabyCrawling', 'BlowingCandles']

#### Create the file list from the directory where the dataset's videos are, and make it your working directory. ####
file_list = os.listdir(r'/<your_file_directory>/UCF101/')
os.chdir(r'/<your_file_directory>/UCF101/')

#### List containing the ground truth classification predictions. ####
train_resp = []
#### List containing the video group in which each video clip belongs. ####
posa = []
#### Counting the video groups. ####
count = -1

#os.chdir(r'/media/spareman/BACKUP/FILES/Ptuxiakh/UCF101/')
for v in range(len(videos)):
	for i in range(1,26):
		count += 1
		for j in range(1,8):
			name = 'v_%s_g%02d_c%02d' % (videos[v], i, j)
			print (name)
			if ('%s.avi' % (name)) not in file_list:
				break
			else:
				posa.append(count)
				train_resp.append(v)

#### Loading the extracted features. ####
fo = open("Final_sift_bag.txt", "rb")
all_sift = np.loadtxt(fo, dtype='float32')
fo.close()
fo = open("Final_mfcc_bag.txt", "rb")
all_mfcc = np.loadtxt(fo, dtype='float32')
fo.close()
fo = open("Final_stip_bag.txt", "rb")
all_stip = np.loadtxt(fo, dtype='float32')
fo.close()
#### Creating the early fusion matrix. ####
all_feat = np.hstack((all_sift, all_mfcc, all_stip))

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn import model_selection

from mlxtend.feature_selection import ColumnSelector
col_sift = ColumnSelector(cols=range(0,2500))
col_mfcc = ColumnSelector(cols=range(2500,6500))
col_stip = ColumnSelector(cols=range(6500,10500))

clsf1 = OneVsRestClassifier(svm.SVC(kernel='precomputed', C=100.0, probability=True, class_weight="balanced"), -1)
clsf2 = OneVsRestClassifier(svm.SVC(kernel='precomputed', C=100.0, probability=True, class_weight="balanced"), -1)
clsf3 = OneVsRestClassifier(svm.SVC(kernel='precomputed', C=100.0, probability=True, class_weight="balanced"), -1)
clsf4 = OneVsRestClassifier(svm.SVC(kernel='precomputed', C=1000.0, probability=True, class_weight="balanced"), -1)

#### Creating the pipelines for each feature type and their early fusion. ####
#### The C, gamma parameters can be found through GridCVSearch. ####
from sklearn.pipeline import Pipeline
sift_pipe = Pipeline([('sel', col_sift), ('chi2', Chi2Kernel(gamma=0.001)),('svm', clsf1)])
mfcc_pipe = Pipeline([('sel', col_mfcc), ('chi2', Chi2Kernel(gamma=0.0001)),('svm', clsf2)])
stip_pipe = Pipeline([('sel', col_stip), ('chi2', Chi2Kernel(gamma=0.01)),('svm', clsf3)])
all_pipe = Pipeline([('chi2', Chi2Kernel(gamma=0.0001)),('svm', clsf4)])

#### Computing baseline accuracies using single feature type each time and then using their early fusion. ####
res1 = cross_val_score(sift_pipe, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res2 = cross_val_score(mfcc_pipe, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res3 = cross_val_score(stip_pipe, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res4 = cross_val_score(all_pipe, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)

#### Extracting the probabilities from each classifier for the late fusion ####
from sklearn.model_selection import cross_val_predict
pro1 = cross_val_predict(sift_pipe, all_feat, train_resp, groups=posa, cv=5, method="predict_proba", n_jobs=-1)
pro2 = cross_val_predict(mfcc_pipe, all_feat, train_resp, groups=posa, cv=5, method="predict_proba", n_jobs=-1)
pro3 = cross_val_predict(stip_pipe, all_feat, train_resp, groups=posa, cv=5, method="predict_proba", n_jobs=-1)
pro4 = cross_val_predict(all_pipe, all_feat, train_resp, groups=posa, cv=5, method="predict_proba", n_jobs=-1)

all_res1 = np.array([pro1, pro2, pro3])
all_res2 = np.array([pro1, pro2, pro3, pro4])

#### Computing final predictions with late fusion without training. ####
#### You can use different combinations of the classifiers to be fused. ####
results11 = all_res1.sum(0).argmax(1)
results12 = all_res1.prod(0).argmax(1)
results13 = np.median(all_res1, 0).argmax(1)
results14 = np.max(all_res1, 0).argmax(1)

#### Computing final predictions with late fusion without training, adding the early fusion classifier. ####
#### You can use different combinations of the classifiers to be fused. ####
results21 = all_res2.sum(0).argmax(1)
results22 = all_res2.prod(0).argmax(1)
results23 = np.median(all_res2, 0).argmax(1)
results24 = np.max(all_res2, 0).argmax(1)

#### You can compute accuracy or a full classification report using the code below, changing the results you want to examine. ####
from sklearn.metrics import accuracy_score
print(accuracy_score(train_resp, results11))
from sklearn.metrics import classification_report
print(classification_report(train_resp, results11, target_names=videos))

#### Creating Majority Voting and Weighted Majority Voting Classifiers with and without using the early fusion classfier. ####
#### You can use different combinations of the classifiers to be fused. ####
from mlxtend.classifier import EnsembleVoteClassifier
eclf1 = EnsembleVoteClassifier(clfs=[sift_pipe, mfcc_pipe, stip_pipe], voting='hard')
eclf2 = EnsembleVoteClassifier(clfs=[sift_pipe, mfcc_pipe, stip_pipe, all_pipe], voting='hard')
weights1 = [res1.mean(), res2.mean(), res3.mean()]
weights2 = [res1.mean(), res2.mean(), res3.mean(), res4.mean()]
weclf1 = EnsembleVoteClassifier(clfs=[sift_pipe, mfcc_pipe, stip_pipe], voting='hard', weights=weights1)
weclf2 = EnsembleVoteClassifier(clfs=[sift_pipe, mfcc_pipe, stip_pipe, all_pipe], voting='hard', weights=weights2)

#### Computing accuracies. ####
res5 = cross_val_score(eclf1, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res6 = cross_val_score(eclf2, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res7 = cross_val_score(weclf1, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res8 = cross_val_score(weclf2, all_feat, train_resp, groups=posa, cv=5, scoring='accuracy', n_jobs=-1)

#### Creating the Stacking Classifier with of without using the early fusion classifier. ####
#### You can use different combinations of the classifiers to be fused. ####
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.01)
from mlxtend.classifier import StackingCVClassifier
sclf1 = StackingCVClassifier(classifiers=[sift_pipe, mfcc_pipe, stip_pipe], meta_classifier=lr, use_probas=True, cv=5)
sclf2 = StackingCVClassifier(classifiers=[sift_pipe, mfcc_pipe, stip_pipe, all_pipe], meta_classifier=lr, use_probas=True, cv=5)
res9 = cross_val_score(sclf1, all_feat, np.array(train_resp), groups=posa, cv=5, scoring='accuracy', n_jobs=-1)
res10 = cross_val_score(sclf2, all_feat, np.array(train_resp), groups=posa, cv=5, scoring='accuracy', n_jobs=-1)

#### You can fine tune the hyper parameters of the classifiers, using code like below. ####
from sklearn.model_selection import GridSearchCV
cv_params = dict([
	('chi2__gamma', 10.0 ** np.arange(-2, 1)),
	('svm__estimator__C', 10.0 ** np.arange(0, 2))
])

grid = GridSearchCV(estimator=sift_pipe, param_grid=cv_params, cv=3, verbose=2, n_jobs=12)
grid.fit(all_feat, train_resp, groups=posa)

#### Use this to find the parameters' names for the cv_params dictionary. ####
pipe.get_params().keys()

#### To plot the confusion matrix, follow the instructions here: 
#### http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html