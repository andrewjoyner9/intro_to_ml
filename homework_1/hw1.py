def run_and_plot(X, y, feature_name):
	theta0, theta1, loss_history = gradient_descent(X, y, lr=0.1, epochs=1000)
	print(f"Feature: {feature_name}")
	print(f"Model: y = {theta0:.4f} + {theta1:.4f}*{feature_name}")
	print(f"Final loss: {loss_history[-1]:.4f}\n")
	plt.figure(figsize=(12,4))
	plt.subplot(1,2,1)
	plt.scatter(X, y, label='Data')
	plt.plot(X, theta0 + theta1*X, color='red', label='Regression line')
	plt.xlabel(feature_name)
	plt.ylabel('Y')
	plt.title(f'Linear Regression ({feature_name}, lr=0.1)')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(loss_history)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Loss over Iterations')
	plt.tight_layout()
	plt.show()

# Run for X1, X2, X3
run_and_plot(X1, y, 'X1')
run_and_plot(X2, y, 'X2')
run_and_plot(X3, y, 'X3')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/HamedTabkhi/Intro-to-ML/main/Dataset/D3.csv"
df = pd.read_csv(url)

X1 = df.values[:,0]
X2 = df.values[:,1]
X3 = df.values[:,2]
y = df.values[:,3]

def gradient_descent(X, y, lr=0.1, epochs=1000):
	m = len(y)
	theta0 = 0.0
	theta1 = 0.0
	loss_history = []
	for i in range(epochs):
		y_pred = theta0 + theta1 * X
		error = y_pred - y
		loss = (1/(2*m)) * np.sum(error ** 2)
		loss_history.append(loss)
		# Gradients
		d_theta0 = (1/m) * np.sum(error)
		d_theta1 = (1/m) * np.sum(error * X)
		# Update
		theta0 -= lr * d_theta0
		theta1 -= lr * d_theta1
	return theta0, theta1, loss_history