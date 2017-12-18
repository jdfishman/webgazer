#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import csv, cv2
import sklearn.svm as svm
import sklearn.linear_model as lm
import sklearn.preprocessing as prep
import numpy as np
from kalmanfilter import KalmanFilter

trainFile = "train_1430_1.txt" 
testFile = "test_1430_1.txt"

ridgeParameter = 10**-5
resizeWidth = 10
resizeHeight = 6

leftKalman = KalmanFilter()
rightKalman = KalmanFilter()


def main():
	poly = prep.PolynomialFeatures(degree=2)
	
	print('STARTING TRAINING')
	#feats, x_labels, y_labels = extract_train_data()
	feats, x_labels, y_labels = load_train_data()

	feats = poly.fit_transform(feats)

	x_model, y_model = train_ridge(feats, x_labels, y_labels)
	#x_model, y_model = train_bayesian_ridge(feats, x_labels, y_labels)
	#x_model, y_model = train_lasso(feats, x_labels, y_labels)
	#x_model, y_model = train_svm(feats, x_labels, y_labels)
	#x_model, y_model = train_sgd(feats, x_labels, y_labels)

	print('STARTING TESTING')
	#feats, x_labels, y_labels = extract_test_data()
	feats, x_labels, y_labels = load_test_data()
	feats = poly.fit_transform(feats)
	test_model(x_model, y_model, feats, x_labels, y_labels)

def extract_train_data(tobii=True):
	return extract_data(True, tobii)

def extract_test_data(tobii=True):
	return extract_data(False, tobii)

def extract_data(train, tobii=True):
	feats = []
	labelsX = []
	labelsY = []

	if train:
		file_path = trainFile
	else:
		file_path = testFile

	with open(file_path, 'r') as f:
		for dir_ in f.readlines():
			#leftKalman.reset()
			#rightKalman.reset()

			dir_ = dir_[:-1]
			csv_path = dir_ + '/gazePredictions.csv'
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

	# save data to file
	dataset = 'train' if train else 'test'
	label_type = 'tobii' if tobii else 'webgazer'
	np.array(feats).dump('%s_features.npy' % (dataset))
	np.array(labelsX).dump('%s_%s_x_labels.npy' % (dataset, label_type))
	np.array(labelsY).dump('%s_%s_y_labels.npy' % (dataset, label_type))

	return feats, labelsX, labelsY

def load_train_data(tobii=True):
	return load_data(True, tobii)

def load_test_data(tobii=True):
	return load_data(False, tobii)

def load_data(train, tobii=True):
	dataset = 'train' if train else 'test'
	label_type = 'tobii' if tobii else 'webgazer'

	return [np.load('%s_features.npy' % (dataset)),
		np.load('%s_%s_x_labels.npy' % (dataset, label_type)),
		np.load('%s_%s_y_labels.npy' % (dataset, label_type))]

def train_ridge(feats, x_labels, y_labels):
	x_model = lm.Ridge(alpha=ridgeParameter).fit(feats,x_labels)
	y_model = lm.Ridge(alpha=ridgeParameter).fit(feats,y_labels)
	return x_model, y_model

def train_bayesian_ridge(feats, x_labels, y_labels):
	x_model = lm.BayesianRidge().fit(feats,x_labels)
	y_model = lm.BayesianRidge().fit(feats,y_labels)
	return x_model, y_model

def train_lasso(feats, x_labels, y_labels):
	x_model = lm.Lasso().fit(feats,x_labels)
	y_model = lm.Lasso().fit(feats,y_labels)
	return x_model, y_model

def train_svm(feats, x_labels, y_labels):
	x_model = svm.SVR().fit(feats,x_labels)
	y_model = svm.SVR().fit(feats,y_labels)
	return x_model, y_model

def train_sgd(feats, x_labels, y_labels):
	x_model = lm.SGDRegressor(tol=1e-3).fit(feats,x_labels)
	y_model = lm.SGDRegressor(tol=1e-3).fit(feats,y_labels)
	return x_model, y_model	

def test_model(x_model, y_model, feats, x_labels, y_labels):
	print('Score-X: ', x_model.score(feats,x_labels))
	print('Score-Y: ', y_model.score(feats,y_labels))
	print('Test Error: ', np.mean(np.sqrt((x_model.predict(feats) - x_labels)**2 + (y_model.predict(feats) - y_labels)**2)))

def getEyeFeats(image_path, clmfeatures):
	global leftKalman, rightKalman

	#Apply Kalman Filtering
	leftBox = [clmfeatures[23][0], clmfeatures[24][1], clmfeatures[25][0], clmfeatures[26][1]]
	#leftBox = leftKalman.update(leftBox)
	leftBox = [int(round(max(x,0))) for x in leftBox]

	#Apply Kalman Filtering
	rightBox = [clmfeatures[30][0], clmfeatures[29][1], clmfeatures[28][0], clmfeatures[31][1]]
	#rightBox = rightKalman.update(rightBox)
	rightBox = [int(round(max(x,0))) for x in rightBox]

	if leftBox[2] - leftBox[0] == 0 or rightBox[2] - rightBox[0] == 0:
		print('an eye patch had zero width')
		return None

	if leftBox[3] - leftBox[1] == 0 or rightBox[3] - rightBox[1] == 0:
		print('an eye patch had zero height')
		return None

	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		return None
	leftImageData = image[leftBox[1]:leftBox[3],leftBox[0]:leftBox[2]]
	rightImageData = image[rightBox[1]:rightBox[3],rightBox[0]:rightBox[2]]

	try:
		resizedLeft = cv2.resize(leftImageData, (resizeWidth, resizeHeight))
		resizedRight = cv2.resize(rightImageData, (resizeWidth, resizeHeight))

		histLeft = cv2.equalizeHist(resizedLeft)
		histRight = cv2.equalizeHist(resizedRight)

		features = np.concatenate((histLeft.flatten(), histRight.flatten()))
		return features

	except:
		return None


if __name__ == '__main__':
	main()