import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv("linearX.csv").values
y=pd.read_csv("linearY.csv").values
X=(X-np.mean(X))/np.std(X)
y=(y-np.mean(y))/np.std(y)
X=np.c_[np.ones(X.shape[0]),X]
def gradient_descent(X, y, alpha, iterations, convergence_criteria=1e-6):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y.flatten()
        cost = (1 / (2 * m)) * np.sum(errors**2)
        cost_history.append(cost)
        gradient = (1 / m) * X.T.dot(errors)
        theta -= alpha * gradient
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < convergence_criteria:
            break
    return theta, cost_history
alpha = 0.5
iterations = 1000
theta, cost_history = gradient_descent(X, y, alpha, iterations)
print("Theta (parameters):", theta)
print("Final cost:", cost_history[-1])
iterations = len(cost_history[:50])  
plt.plot(range(iterations), cost_history[:iterations])
plt.title("Cost Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
plt.scatter(X[:, 1], y, color='blue', label='Dataset')
plt.plot(X[:, 1], X.dot(theta), color='red', label='Regression Line')
plt.title("Dataset and Regression Line")
plt.xlabel("Predictor Variable")
plt.ylabel("Response Variable")
plt.legend()
plt.show()
learning_rates = [0.005, 0.5, 5]
for lr in learning_rates:
    _, cost_hist = gradient_descent(X, y, lr, 50)
    plt.plot(range(len(cost_hist)), cost_hist, label=f"lr={lr}")
plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()
def stochastic_gradient_descent(X, y, alpha, iterations):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []
    for i in range(iterations):
        for j in range(m):
            rand_idx = np.random.randint(0, m)
            X_sample = X[rand_idx, :].reshape(1, -1)
            y_sample = y[rand_idx]
            
            prediction = X_sample.dot(theta)
            error = prediction - y_sample
            theta -= alpha * error * X_sample.flatten()
        cost = (1 / (2 * m)) * np.sum((X.dot(theta) - y.flatten())**2)
        cost_history.append(cost)
    return theta, cost_history
def mini_batch_gradient_descent(X, y, alpha, iterations, batch_size):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []
    for i in range(iterations):
        for j in range(0, m, batch_size):
            X_batch = X[j:j + batch_size, :]
            y_batch = y[j:j + batch_size]
            predictions = X_batch.dot(theta)
            errors = predictions - y_batch.flatten()
            theta -= alpha * (1 / batch_size) * X_batch.T.dot(errors)
        cost = (1 / (2 * m)) * np.sum((X.dot(theta) - y.flatten())**2)
        cost_history.append(cost)
    return theta, cost_history
theta_sgd, cost_sgd = stochastic_gradient_descent(X, y, alpha=0.05, iterations=50)
theta_mbgd, cost_mbgd = mini_batch_gradient_descent(X, y, alpha=0.05, iterations=50, batch_size=16)
plt.plot(range(len(cost_history)), cost_history, label="Batch Gradient Descent")
plt.plot(range(len(cost_sgd)), cost_sgd, label="Stochastic Gradient Descent")
plt.plot(range(len(cost_mbgd)), cost_mbgd, label="Mini-Batch Gradient Descent")
plt.title("Cost Function vs Iterations for Different Methods")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()