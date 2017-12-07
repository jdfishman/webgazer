import numpy as np

class KalmanFilter:

	def __init__(self, F, H, Q, R, P_initial, X_initial):
		self.F = F
		self.H = H
		self.Q = Q
		self.R = R
		self.P = P_initial
		self.X = X_initial

	def update(self, z):
		# TODO cache variables like the transpose of H

		#  prediction: X = F * X  |  P = F * P * F' + Q
		X_p = np.matmul(self.F, self.X) # Update state vector
		P_p = np.matmul(np.matmul(self.F,self.P), np.transpose(self.F)) + self.Q # Predicted covaraince

		# Calculate the update values
		y = z - np.matmul(self.H, X_p) #  This is the measurement error (between what we expect and the actual value)
		S = np.matmul(np.matmul(self.H, P_p), np.transpose(self.H)) + self.R # This is the residual covariance (the error in the covariance)

		#  kalman multiplier: K = P * H' * (H * P * H' + R)^-1
		K = np.matmul(P_p, np.matmul(np.transpose(self.H), np.linalg.pinv(S))) # This is the Optimal Kalman Gain

		# We need to change Y into it's column vector form
		y = np.transpose(y)

		# Now we correct the internal values of the model
		#  correction: X = X + K * (m - H * X)  |  P = (I - K * H) * P
		self.X = X_p + np.matmul(K, y)
		self.P = np.matmul(np.identity(len(K)) - np.matmul(K,self.H), P_p)
		return np.transpose(np.matmul(self.H, self.X))[0] # Transforms the predicted state back into it's measurement form

