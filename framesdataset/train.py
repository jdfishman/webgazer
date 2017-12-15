#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import os, glob, random, csv, cv2, scipy.linalg
import sklearn.svm as svm
import numpy as np
from kalmanfilter import KalmanFilter

trainFile = "train_1430_1.txt" 
testFile = "test_1430_1.txt"

ridgeParameter = 10**-5
resizeWidth = 10
resizeHeight = 6
alpha = 0.5 # 0.5-1

leftKalman = None
rightKalman = None


def main():
	init_kalman_filters()

	# COLLECT TRAINING DATA -----------------
	print('COLLECTING DATA')
	feats, x_labels, y_labels = extract_data()
	feats, x_labels, y_labels = load_data()


	
	print('STARTING TRAINING')
	x_svm = svm.SVR()
	y_svm = svm.SVR()
	# TRAIN WEIGHTS -------------------------
	if len(feats) > 0:
		x_svm.fit(feats,labelsX)
		y_svm.fit(feats,labelsY)
		print(x_svm.get_params())
		print(y_svm.get_params())
		# x_weights = ridge(feats, labelsX, ridgeParameter)
		# y_weights = ridge(feats, labelsY, ridgeParameter)
		# print('X_WEIGHTS: ', x_weights)
		# print('Y_WEIGHTS: ', y_weights)
		print('Score-X: ', x_svm.score(feats,labelsX))
		print('Score-Y: ', y_svm.score(feats,labelsY))
		print('Error: :', np.mean(np.sqrt((x_svm.predict(feats) - labelsX)**2 + (y_svm.predict(feats) - labelsY)**2)))

	return x_svm, y_svm

	# x_weights = np.array([ -1.59931874e-04,   1.69764108e-04,   3.19109954e-05,
 #         3.51339963e-04,   4.61521250e-04,  -1.21235509e-03,
 #         8.05921415e-05,  -4.04467046e-04,   1.66974689e-04,
 #        -1.93671705e-04,   6.03213842e-05,   4.10648068e-04,
 #        -2.22708531e-04,  -1.16366171e-04,  -6.74957676e-05,
 #         5.48688657e-04,   2.73937282e-04,   1.89243182e-04,
 #         2.93250995e-05,   5.57202889e-04,  -7.66679262e-05,
 #        -2.44862674e-04,   3.78131601e-05,   5.07354844e-05,
 #        -1.90610700e-05,  -2.23481188e-04,   1.46531427e-04,
 #         1.07270124e-05,  -3.91064917e-04,  -1.94727489e-04,
 #        -4.43718203e-05,   3.17228062e-04,   1.57504190e-04,
 #        -2.00780725e-05,  -2.59324495e-04,   2.47273333e-04,
 #         4.04818700e-04,  -3.34142864e-04,  -6.14756439e-05,
 #         3.44454944e-05,  -1.00084089e-04,  -1.10228530e-04,
 #        -1.43138057e-04,   1.71084094e-04,   9.20696815e-05,
 #        -3.11522141e-04,  -3.76471932e-04,   1.95333686e-04,
 #         1.08642996e-05,   1.05285416e-04,  -1.22793924e-04,
 #        -9.01634048e-05,  -1.30526773e-04,   3.92334888e-05,
 #         6.59229550e-05,   1.36823748e-05,  -2.57329970e-05,
 #        -2.21485025e-04,  -1.86466089e-04,   1.22655506e-04,
 #         1.41778303e-04,  -9.52643554e-05,  -6.80522769e-05,
 #         6.69021450e-05,  -6.01781816e-05,   1.01800956e-04,
 #         1.00442253e-04,   2.57554856e-04,   2.09441459e-04,
 #        -9.83740462e-05,   1.64547108e-04,   4.76563097e-04,
 #        -3.14537425e-04,   4.58856188e-04,  -3.93805941e-05,
 #         1.39481182e-04,   3.91814343e-04,  -2.44003650e-04,
 #        -2.65778879e-05,  -1.37544619e-04,   1.21245687e-04,
 #         6.48585697e-04,   9.69721334e-05,   3.14238360e-04,
 #         2.02891243e-04,  -1.22093747e-04,   4.96169491e-04,
 #        -1.53869945e-04,  -2.32148487e-04,  -3.22162767e-06,
 #        -2.86472305e-04,  -1.58119006e-04,  -2.15425877e-05,
 #        -1.61203569e-05,   2.63125820e-06,   2.17447079e-04,
 #         2.91946709e-04,   1.76377971e-04,   2.50464030e-04,
 #         1.19101729e-04,  -2.23674730e-04,  -1.60853943e-04,
 #        -2.89540898e-04,  -1.22547663e-04,   1.73994660e-04,
 #         4.75806095e-04,  -6.75504284e-05,   8.37513830e-05,
 #         2.36302714e-04,  -1.70119459e-04,   3.53134177e-04,
 #         1.42584762e-05,  -2.39670948e-04,   1.72153126e-04,
 #         5.83491597e-04,   4.35608160e-04,  -1.84043488e-04,
 #         1.36684977e-04,  -1.79874301e-04,  -2.67845047e-05,
 #         2.39571915e-07])
	# y_weights = np.array([ -3.51432919e-04,  -1.98472690e-04,  -3.29274044e-04,
 #        -1.93668577e-04,   6.92618810e-04,  -9.31960385e-04,
 #         2.15607722e-04,  -2.31638781e-04,  -1.79245358e-04,
 #        -9.53421599e-05,  -6.81766889e-05,   1.81551206e-04,
 #         2.39307625e-05,  -2.15536112e-04,   1.35140221e-04,
 #        -9.96575918e-05,   2.19326102e-04,   1.34140475e-04,
 #         1.16364594e-04,   3.09237440e-04,   7.12974289e-05,
 #         4.56737910e-05,   1.99773046e-04,  -1.86194295e-04,
 #         2.14961778e-04,  -1.74633079e-04,   1.00099438e-04,
 #        -1.02202221e-04,  -3.43099456e-05,  -4.07762208e-04,
 #         1.33575220e-04,   5.08973958e-04,   2.27481952e-04,
 #         1.43897011e-04,  -2.36972007e-04,   2.69425768e-04,
 #         2.71306878e-04,  -1.40289660e-04,  -1.36786762e-04,
 #        -2.11630258e-04,  -1.13804054e-04,  -1.16642169e-04,
 #        -2.17834619e-04,   2.72839483e-04,   6.48627030e-05,
 #        -4.33174905e-05,   5.95847998e-05,   1.26834657e-05,
 #         4.99185702e-05,  -1.11131997e-04,   8.18369167e-05,
 #        -3.23491278e-04,  -8.60112824e-05,  -7.29568789e-05,
 #         1.70275948e-06,  -1.74992122e-04,  -3.74006032e-04,
 #        -1.55051825e-04,  -6.67253092e-05,  -1.18664753e-04,
 #         4.11656648e-04,   2.62028307e-04,   4.56918703e-05,
 #         9.32819635e-04,   1.44519103e-04,   1.51784354e-05,
 #        -1.40508293e-06,   2.72024766e-04,   1.13149984e-04,
 #        -5.56848721e-05,   1.97545304e-04,   6.26174602e-05,
 #         2.78774744e-04,  -2.50501075e-04,   5.13626627e-04,
 #        -2.59003554e-04,  -1.31837720e-04,   1.17398315e-04,
 #        -3.64495115e-06,  -1.40011721e-04,   4.85195914e-04,
 #         2.75784293e-05,   6.23065847e-05,   2.11904786e-05,
 #         8.87820241e-05,  -3.11221921e-05,   3.87514516e-04,
 #        -1.25875681e-04,  -1.97159533e-04,  -1.43726282e-04,
 #        -2.12790721e-04,  -1.54645999e-04,   7.43858263e-05,
 #        -2.17131861e-04,  -1.76081402e-04,  -2.63124039e-04,
 #         7.17467798e-05,   2.64451115e-04,  -5.01349036e-05,
 #         1.47801052e-04,  -3.69966681e-05,   7.27105399e-05,
 #         2.33924350e-04,   1.47417927e-04,   1.37060635e-04,
 #         5.33375941e-04,   2.29160526e-04,  -1.18320690e-04,
 #        -3.87418859e-05,   1.34522543e-04,   2.28449176e-04,
 #        -5.62974333e-05,   1.17045898e-04,  -1.54343733e-04,
 #         2.75393500e-04,   2.98529326e-04,   3.14821579e-04,
 #         3.26253274e-04,   1.37987668e-04,   1.87578811e-04,
 #         8.90769155e-07])

	# CALCULATE TEST ERROR ------------------
	# with open( testFile, 'r' ) as file:
	# 	a=0
	# 	total_error = 0
	# 	for train_dir in file.readlines():

	# 		train_dir = train_dir[:-1]
	# 		csv_path = train_dir + '/gazePredictions.csv'
	# 		with open(csv_path, 'r') as csv_file:
	# 			gaze_csv = csv.reader(csv_file, delimiter=',')

	# 			for row in gaze_csv:
	# 				image_path = row[0]
	# 				labelX = float(row[6])
	# 				labelY = float(row[7])
	# 				clm_data = np.array([float(x) for x in row[8:len(row)-1]])
	# 				clm_data = clm_data.reshape(-1,2)

	# 				eye_feats = getEyeFeats(image_path, clm_data)

	# 				if eye_feats is not None:

	# 					# #guess labels
	# 					#guessX = np.dot(x_weights, np.append(eye_feats,[1]))
	# 					#guessY = np.dot(y_weights, np.append(eye_feats,[1]))
	# 					#error = np.sqrt((guessX-labelX)**2 + (guessY-labelY)**2)
	# 					# print('Guess: ', guessX, ', ', guessY)
	# 					# print('Label: ', labelX, ', ', labelY)
	# 					score1 = x_svm.score()
	# 					a=a+1
	# 					total_error += error

	# 	print('Average error:', total_error/a)

def extract_data(tobii=True):
	feats = []
	labelsX = []
	labelsY = []

	# COLLECT TRAINING DATA -----------------
	with open( trainFile, 'r' ) as trf:
		for train_dir in trf.readlines():

			train_dir = train_dir[:-1]
			csv_path = train_dir + '/gazePredictions.csv'
			with open(csv_path, 'r') as csv_file:
				gaze_csv = csv.reader(csv_file, delimiter=',')
				
				for row in gaze_csv:
					image_path = row[0]
					if tobii:
						tobiiLeftEyeGazeX = float( row[2] )
						tobiiLeftEyeGazeY = float( row[3] )
						tobiiRightEyeGazeX = float( row[4] )
						tobiiRightEyeGazeY = float( row[5] )
						labelX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
						labelY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2
					else:
						labelX = float(row[6])
						labelY = float(row[7])
					clm_data = np.array([float(x) for x in row[8:len(row)-1]])
					clm_data = clm_data.reshape(-1,2)

					eye_feats = getEyeFeats(image_path, clm_data)

					if eye_feats is not None:

						eye_feats = np.append(eye_feats,[1])
						feats.append(eye_feats)
						labelsX.append(labelX)
						labelsY.append(labelY)

	np.array(feats).dump('features.npy')
	np.array(labelsX).dump('%s_x_labels.npy'.format('tobii' if tobii else 'webgazer'))
	np.array(labelsY).dump('%s_y_labels.npy'.format('tobii' if tobii else 'webgazer'))

	return feats, labelsX, labelsY

def load_data(tobii=True):
	return np.load('features.npy'),
		np.load('%s_x_labels.npy'.format('tobii' if tobii else 'webgazer')),
		np.load('%s_y_labels.npy'.format('tobii' if tobii else 'webgazer'))


def ridge(X,y,k):
	A = np.matmul(np.transpose(X), X) + k * np.identity(len(X[0]))
	b = np.matmul(np.transpose(X), y)

	weights = scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), b)
	#weights = np.matmul(np.linalg.inv(A), b)
	return weights

def online_ridge(X,y,n,weights):
	update = X * (y - np.transpose(X)*weights)
	weights = weights + (n ** -alpha) * update
	return weights


def init_kalman_filters():
	global leftKalman, rightKalman

	F = np.array([ [1, 0, 0, 0, 1, 0],
					[0, 1, 0, 0, 0, 1],
					[0, 0, 1, 0, 1, 0],
					[0, 0, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 1]])
	#Parameters Q and R may require some fine tuning
	Q = np.array([ [1/4,  0, 0, 0,  1/2,   0],
					[0, 1/4,  0, 0,    0, 1/2],
					[0, 0,   1/4, 0, 1/2,   0],
					[0, 0,   0,  1/4,  0, 1/2],
					[1/2, 0, 1/2, 0,    1,  0],
					[0, 1/2,  0,  1/2,  0,  1]])
	delta_t = 1/10 # The amount of time between frames
	Q = Q * delta_t
	H = np.array([ [1, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0],
					[0, 0, 0, 1, 0, 0]])
	pixel_error = 6.5 # We will need to fine tune this value
	#This matrix represents the expected measurement error
	R = pixel_error * np.identity(4)

	P_initial = 0.0001 * np.identity(6) #Initial covariance matrix
	x_initial = np.array([[200], [150], [250], [180], [0], [0]]) # Initial measurement matrix

	leftKalman = KalmanFilter(F, H, Q, R, P_initial, x_initial)
	rightKalman = KalmanFilter(F, H, Q, R, P_initial, x_initial)

def getEyeFeats(image_path, clmfeatures):
	global leftKalman, rightKalman

	# Fit the detected eye in a rectangle
	leftOriginX = (clmfeatures[23][0])
	leftOriginY = (clmfeatures[24][1])
	leftWidth = (clmfeatures[25][0] - clmfeatures[23][0])
	leftHeight = (clmfeatures[26][1] - clmfeatures[24][1])
	rightOriginX = (clmfeatures[30][0])
	rightOriginY = (clmfeatures[29][1])
	rightWidth = (clmfeatures[28][0] - clmfeatures[30][0])
	rightHeight = (clmfeatures[31][1] - clmfeatures[29][1])

	#Apply Kalman Filtering
	leftBox = [leftOriginX, leftOriginY, leftOriginX + leftWidth, leftOriginY + leftHeight]
	#leftBox = leftKalman.update(leftBox)
	leftBox = [int(round(max(x,0))) for x in leftBox]

	#Apply Kalman Filtering
	rightBox = [rightOriginX, rightOriginY, rightOriginX + rightWidth, rightOriginY + rightHeight]
	#rightBox = rightKalman.update(rightBox)
	rightBox = [int(round(max(x,0))) for x in rightBox]

	if leftWidth == 0 or rightWidth == 0:
		print('an eye patch had zero width')
		return None

	if leftHeight == 0 or rightHeight == 0:
		print('an eye patch had zero height')
		return None

	#eyeObjs = {}
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		return None
	leftImageData = image[leftBox[1]:leftBox[3],leftBox[0]:leftBox[2]]
	#print((leftBox[2]-leftBox[0], leftBox[3] - leftBox[1]))
	#eyeObjs['left'] = 
	# {
	# 		'patch': leftImageData,
	# 		'imagex': leftOriginX,
	# 		'imagey': leftOriginY,
	# 		'width': leftWidth,
	# 		'height': leftHeight
	# }

	rightImageData = image[rightBox[1]:rightBox[3],rightBox[0]:rightBox[2]]
	#eyeObjs['right'] = 
	# {
	# 		'patch': rightImageData,
	# 		'imagex': rightOriginX,
	# 		'imagey': rightOriginY,
	# 		'width': rightWidth,
	# 		'height': rightHeight
	# }

	#eyeObjs['positions'] = clmfeatures

	try:

		resizedLeft = cv2.resize(leftImageData, (resizeWidth, resizeHeight))
		resizedRight = cv2.resize(rightImageData, (resizeWidth, resizeHeight))

		# leftGray = cv2.cvtColor(resizedLeft, cv2.COLOR_RGB2GRAY)
		# rightGray = cv2.cvtColor(resizedRight, cv2.COLOR_RGB2GRAY)

		histLeft = cv2.equalizeHist(resizedLeft)
		histRight = cv2.equalizeHist(resizedRight)

		features = np.concatenate((histLeft.flatten(), histRight.flatten()))
		return features

	except:

		return None



if __name__ == '__main__':
	main()