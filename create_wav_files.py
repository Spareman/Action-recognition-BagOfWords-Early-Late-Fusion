import os
import time
import subprocess

#### Class names you want to extract wav files from. ####
videos = ['ApplyEyeMakeup', 'SoccerPenalty', 'BabyCrawling', 'Typing', 'BlowingCandles', 'Haircut',
			'FloorGymnastics', 'BrushingTeeth', 'FieldHockeyPenalty', 'HeadMassage', 'ParallelBars', 'HandstandWalking']

n_filters = 40              # must be 40 for mfcc
n_coeffs = 13

#### Create the file list from the directory where the dataset's videos are, and make it your working directory. ####
file_list = os.listdir(r'/<your_file_directory>/UCF101/')
os.chdir(r'/<your_file_directory>/UCF101/')


#### Process all videos from the classes in the videos list. ####
for video in videos:
	for i in range(1,26):
		for j in range(1,8):
			name = 'v_%s_g%02d_c%02d' % (video, i, j)
			if (name + '.avi') in file_list:
				command = "ffmpeg -i %s.avi -vn %s.wav" % (name, name)
				response = subprocess.call(command, shell=True)
#### The wav files are saved in the working directory. ####				
				while (response != 0):
					print ("Not yet")
					time.sleep(2)

			else:
				break
				
print ("All ended well")