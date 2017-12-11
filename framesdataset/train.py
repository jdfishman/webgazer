#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import os, glob, random, csv, cv2, scipy.linalg
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
	# x_weights = np.random.normal(0,.001,resizeWidth*resizeHeight*2+1)
	# y_weights = np.random.normal(0,.001,resizeWidth*resizeHeight*2+1)
	# n = 1

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

						# #guess labels
						# guessX = np.dot(x_weights, eye_feats)
						# guessY = np.dot(y_weights, eye_feats)
						# error = np.sqrt((guessX-labelX)**2 + (guessY-labelY)**2)
						# print('Guess: ', guessX, ', ', guessY)
						# print('Label: ', labelX, ', ', labelY)
						# print('Error: ', error)

						# # update weights
						# x_weights = online_ridge(eye_feats, labelX, n, x_weights)
						# y_weights = online_ridge(eye_feats, labelY, n, y_weights)
						# n = n+1

	# TRAIN WEIGHTS -------------------------
	if len(feats) > 0:
		x_weights = ridge(feats, labelsX, ridgeParameter)
		y_weights = ridge(feats, labelsY, ridgeParameter)
		print('X_WEIGHTS: ', x_weights)
		print('Y_WEIGHTS: ', y_weights)

	# CALCULATE TEST ERROR ------------------
	with open( testFile, 'r' ) as file:
		a=0
		total_error = 0
		for train_dir in file.readlines():

			train_dir = train_dir[:-1]
			csv_path = train_dir + '/gazePredictions.csv'
			with open(csv_path, 'r') as csv_file:
				gaze_csv = csv.reader(csv_file, delimiter=',')

				for row in gaze_csv:
					image_path = row[0]
					labelX = float(row[6])
					labelY = float(row[7])
					clm_data = np.array([float(x) for x in row[8:len(row)-1]])
					clm_data = clm_data.reshape(-1,2)

					eye_feats = getEyeFeats(image_path, clm_data)

					if eye_feats is not None:

						# #guess labels
						guessX = np.dot(x_weights, np.append(eye_feats,[1]))
						guessY = np.dot(y_weights, np.append(eye_feats,[1]))
						error = np.sqrt((guessX-labelX)**2 + (guessY-labelY)**2)
						# print('Guess: ', guessX, ', ', guessY)
						# print('Label: ', labelX, ', ', labelY)
						a=a+1
						total_error += error

		print('Average error:', total_error/a)



	trf.close()

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