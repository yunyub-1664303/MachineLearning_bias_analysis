import numpy as np 
import math

data = open('titanic_data.csv').read().split('\n')
data.pop(0)
data.remove("")

x = []
y = []
for entry in data:
  entry = entry.split(',')
  y.append(list(map(float, entry[0]))[0])
  x_i = []
  for i in range(1, len(entry)):
    x_i.append(entry[i])
  x.append(list(map(float, x_i)))
# print(x)
# print(y)

def get_gradient(theta, x, y):
  theta = np.array(theta)
  return sum(np.dot(y[i] - 1/(1+math.exp(theta.dot(x[i]))), np.array(x[i])) for i in range(len(x)))

def get_l(theta, x, y):
  theta = np.array(theta)
  return sum(y[i] * np.log(1/(1+math.exp(-np.dot(theta.transpose(), np.array(x[i]))))) + (1-y[i]) * np.log(1/(1+math.exp(np.dot(theta.transpose(), np.array(x[i]))))) for i in range(len(x)))

step = 0.0001
epsilon = math.pow(10, -8)
theta_t = [1/12, 1/4, 1/200, 1/4, 1/5, 1/20]
gradient = get_gradient(theta_t, x, y)
# print(gradient)
theta_t1 = np.add(theta_t, np.multiply(step, gradient))
iteration = 0
while (theta_t1 - theta_t).all() > epsilon and iteration < 1000000:
  iteration += 1
  theta_t = theta_t1
  theta_t1 = theta_t + step * get_gradient(theta_t, x, y)
  if iteration % 10000 == 0:
    print(iteration)
print(iteration)
print(theta_t1)
print(get_l(theta_t1, x, y))

# 3e, get inverse of fisher info
# theta_hat = np.array([-1.178,2.757,-0.043,-0.402,-0.107,0.003])
# print(get_l(theta_hat, x, y))
# I = []
# for i in range(6):
#   I.append([])
#   for j in range(6):
#     I[i].append(0)
# for i in range(len(x)):
#   coeff = math.exp(-theta_hat @ x[i])/math.pow(1 + math.exp(-theta_hat @ x[i]), 2)
#   for j in range(len(x[i])):
#     for k in range(len(x[i])):
#       I[j][k] += x[i][j] * x[i][k] * coeff
# print(np.linalg.inv(I))
# sum(math.exp(-np.dot(theta_hat.transpose(), x[i]))/math.pow(1 + math.exp(-np.dot(theta_hat.transpose(), x[i])), 2) for i in range(len(x)))
