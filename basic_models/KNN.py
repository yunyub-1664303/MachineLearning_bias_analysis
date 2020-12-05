import math
import heapq
import matplotlib.pyplot as plt 
import numpy as np

# 5.1
class Point:
  def __init__(self, pclass, sex, age, siblings, parents, fare):
    self.pclass = pclass
    self.sex = sex
    self.age = age / age_norm
    self.siblings = siblings
    self.parents = parents
    self.fare = fare / fare_norm
    
  def get_dist(self, other):
    dist = math.sqrt((self.pclass - other.pclass)**2 + (self.sex - other.sex)**2 + (self.age - other.age)**2 
      + (self.siblings - other.siblings)**2 + (self.parents - other.parents)**2 + (self.fare - other.fare)**2)
    return dist

fare_norm = 52
age_norm = 8
N = 887

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

def KNN(new_point, K):
  q = []
  for i in range(len(x)):
    point = Point(x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5])
    heapq.heappush(q, (new_point.get_dist(point), i))
  nsmallest = q[:K]
  # print(nsmallest)
  cnt = 0
  for neighbor in nsmallest:
    cnt += y[neighbor[1]]
  # print(cnt)
  if cnt > K / 2:
    return 1
  else:
    return 0

# x_axis = []
# y_axis = []
# for i in range(887):
#   x_axis.append(i)
#   y_axis.append(KNN(Point(1, 1, 22, 1, 1, 70), i))
# plt.plot(np.array(x_axis), np.array(y_axis))
# plt.show()

# heap test
# q = []
# heapq.heappush(q, 10)
# heapq.heappush(q, 8)
# print(q)

# 5.2
# pclass_0 = []
# pclass_1 = []
# sex_0 = []
# sex_1 = []
# age_0 = []
# age_1 = []
# siblings_0 = []
# siblings_1 = []
# parents_0 = []
# parents_1 = []
# fare_0 = []
# fare_1 = []

# for i in range(len(x)):
#   if y[i] == 0:
#     pclass_0.append(x[i][0])
#     sex_0.append(x[i][1])
#     age_0.append(x[i][2])
#     siblings_0.append(x[i][3])
#     parents_0.append(x[i][4])
#     fare_0.append(x[i][5])
#   else:
#     pclass_1.append(x[i][0])
#     sex_1.append(x[i][1])
#     age_1.append(x[i][2])
#     siblings_1.append(x[i][3])
#     parents_1.append(x[i][4])
#     fare_1.append(x[i][5])

# def get_mean(lst):
#   return sum(lst) / len(lst)

# def get_var(lst, avg):
#   return sum((x-avg)**2 for x in lst) / len(lst)

# def get_conditional_p(feature, x_val):
#   probabilities = []
#   for i in range(x_val + 1):
#     probabilities.append(0)
#   for value in feature:
#     probabilities[int(value)] += 1
#   for i in range(x_val + 1):
#     probabilities[i] /= len(feature)
#   return probabilities

# p_pclass_0 = get_conditional_p(pclass_0, 3)
# p_pclass_1 = get_conditional_p(pclass_1, 3)

# p_sex_0 = get_conditional_p(sex_0, 1)
# p_sex_1 = get_conditional_p(sex_1, 1)

# age_0_avg = get_mean(age_0)
# age_0_var = get_var(age_0, age_0_avg)
# age_1_avg = get_mean(age_1)
# age_1_var = get_var(age_1, age_1_avg)

# p_siblings_0 = get_conditional_p(siblings_0, 8)
# p_siblings_1 = get_conditional_p(siblings_1, 8)

# p_parents_0 = get_conditional_p(parents_0, 6)
# p_parents_1 = get_conditional_p(parents_1, 6)

# fare_0_avg = get_mean(fare_0)
# fare_0_var = get_var(fare_0, fare_0_avg)
# fare_1_avg = get_mean(fare_1)
# fare_1_var = get_var(fare_1, fare_1_avg)

# def get_p_gaussian(val, mean, var):
#   return 1/math.sqrt(2 * math.pi * var) * math.exp(- (val - mean) ** 2 / (2 * var))

# def naive_bayes(vector):
#   p_1 = sum(y) / 887
#   p_1 *= p_pclass_1[vector[0]]
#   p_1 *= p_sex_1[vector[1]]
#   p_1 *= get_p_gaussian(vector[2], age_1_avg, age_1_var)
#   p_1 *= p_siblings_1[vector[3]]
#   p_1 *= p_parents_1[vector[4]]
#   p_1 *= get_p_gaussian(vector[5], fare_1_avg, fare_1_var)

#   p_0 = 1 - p_1
#   p_0 *=  p_parents_0[vector[1]]
#   p_0 *= p_sex_0[vector[1]]
#   p_0 *= get_p_gaussian(vector[2], age_0_avg, age_0_var)
#   p_0 *= p_siblings_0[vector[3]]
#   p_0 *= p_parents_0[vector[4]]
#   p_0 *= get_p_gaussian(vector[5], fare_0_avg, fare_0_var)

#   if p_0 > p_1:
#     return 0
#   else:
#     return 1

# print(naive_bayes([1, 1, 22, 1, 1, 70]))