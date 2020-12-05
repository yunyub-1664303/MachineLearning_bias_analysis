import math
from pptree import *
import copy
import random

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

# problem 4.1 find optimal threshold to binarize variables
pclass = []
sex = []
age = []
siblings = []
parents = []
fare = []

for entry in x:
  pclass.append(entry[0])
  sex.append(entry[1])
  age.append(entry[2])
  siblings.append(entry[3])
  parents.append(entry[4])
  fare.append(entry[5])

# print out the mean and variance of each list
def print_mean_var(lst):
  avg = sum(lst) / len(lst)
  print(avg)
  print(sum((x-avg)**2 for x in lst) / len(lst))
print_mean_var(parents)

# get the mutual information I(x, y) = H(x) - H(x|y)
def get_mutual_info(x, y):
  px_0 = 0
  py_0 = 0
  px_0y_0 = 0
  px_0y_1 = 0
  px_1y_0 = 0
  px_1y_1 = 0
  for i in range(len(x)):
    if x[i] == 0:
      px_0 += 1
    if y[i] == 0:
      py_0 += 1
    if x[i] == 0 and y[i] == 0:
      px_0y_0 += 1
    elif x[i] == 0 and y[i] == 1:
      px_0y_1 += 1
    elif x[i] == 1 and y[i] == 0:
      px_1y_0 += 1
    else:
      px_1y_1 += 1
  px_0 /= len(x)
  py_0 /= len(x)
  px_0y_0 /= len(x)
  px_0y_1 /= len(x)
  px_1y_0 /= len(x)
  px_1y_1 /= len(x)
  if px_0 == 0 or px_0 == 1 or py_0 == 0 or py_0 == 1:
    return 0
  hx = px_0 * math.log2(1/px_0) + (1 - px_0) * math.log2(1/(1 - px_0))
  hxy = 0
  if px_0y_0 != 0:
    hxy += px_0y_0 * math.log2(py_0/px_0y_0)
  if px_0y_1 != 0:
    hxy += px_0y_1 * math.log2((1 - py_0)/px_0y_1)
  if px_1y_0 != 0:
    hxy += px_1y_0 * math.log2(py_0/px_1y_0)
  if px_1y_1 != 0:
    hxy += px_1y_1 * math.log2((1 - py_0)/px_1y_1)
  return hx - hxy

# binarize the given attribute
# start: the smallest possible value of the attribute
# end: the largest possible value of the attribute
# data: list of the attribute values
def binarize(start, end, data):
  I = 0
  # optimal_cutoff = start
  res = []
  for threshold in range(start, end):
    binarized = []
    for i in range(N):
      if data[i] <= threshold:
        binarized.append(0)
      else:
        binarized.append(1)
    info = get_mutual_info(binarized, y)
    if info > I:
      I = info
      # optimal_cutoff = threshold
      res = binarized
  # print(optimal_cutoff)
  return res
pclass = binarize(1, 3, pclass)
age = binarize(1, 80, age)
siblings = binarize(0, 8, siblings)
parents = binarize(0, 6, parents)
fare = binarize(0, 512, fare)

# pclass 1, 2 -> 0
# age <= 6 -> 0
# siblings <= 3 -> 0
# parents == 0 -> 0
# fare <= 11 -> 0

# problem 4.3
# generates the output of the decision tree
# xd: list of list of attribute values
# y: list of outcomes
# parent = None, flag = None
# index_to_features: store the list of attribute names in the same order 
# as they are in xd
def build_decision_tree(xd, y, parent, flag, index_to_feature):
  itf = copy.deepcopy(index_to_feature)
  # step 1
  maxI = 0
  split = 0
  cnt = 0
  for i in range(len(xd)):
    xi = xd[i]
    cnt += len(xi)
    Ixiy = get_mutual_info(xi, y)
    if Ixiy > maxI:
      maxI = Ixiy
      split = i
  # stopping condition
  if maxI == 0 or cnt < 0.05 * N:
    y0 = 0
    for num in y:
      if num == 0:
        y0 += 1
    if y0 > len(y) / 2:
      return Node(str(flag) + "---0", parent)
    else:
      return Node(str(flag) + "---1", parent)
    
  # step 2
  xd0 = []
  y0 = []
  xd1 = []
  y1 = []
  for i in range(len(xd)):
    if i == split:
      continue
    xd0i = []
    xd1i = []
    for j in range(len(xd[i])):
      if xd[split][j] == 0:
        xd0i.append(xd[i][j])
      else:
        xd1i.append(xd[i][j])
    xd0.append(xd0i)
    xd1.append(xd1i)

  for j in range(len(xd[split])):
    if xd[split][j] == 0:
      y0.append(y[j])
    else:
      y1.append(y[j])
  if parent is None:
    head = Node(itf[split])
  else:
    head = Node(str(flag) + "---" + itf[split], parent)
  itf.remove(index_to_feature[split])
  build_decision_tree(xd0, y0, head, 0, itf)
  build_decision_tree(xd1, y1, head, 1, itf)
  return head

xd = [pclass, sex, age, siblings, parents, fare]
index_to_feature = ["pclass", "sex", "age", "siblings", "parents", "fare"]
# tree = build_decision_tree(xd, y, None, None, index_to_feature)
# print_tree(tree)

# problem 4.5 
# def ten_fold_cross_validation(xd, y, itf):
#   correct = 0
#   for i in range(10):
#     start = i * N//10
#     end = (i+1) * (N//10)
#     trainingx = []
#     for i in range(len(xd)):
#       training_feature = []
#       for j in range(len(xd[i])):
#         if j < start or j > end:
#           training_feature.append(xd[i][j])
#       trainingx.append(training_feature)
#     trainingy = y[:start] + y[end + 1:]

#     tree = build_decision_tree(trainingx, trainingy, None, None, itf)
#     for index in range(i * N//10, (i+1) * (N//10)):
#       true_x = []
#       for i in range(len(xd)):
#         true_x.append(xd[i][index])
#       true_y = y[index]
#       if predict(tree, true_x, true_y, itf):
#         correct += 1
#   return correct / N

# def predict(tree, x, y, itf):
#   while not tree.name.endswith('1') and not tree.name.endswith('0'):
#     names = tree.name.split("---")
#     feature = names[len(names) - 1]
#     val = int(x[itf.index(feature)])
#     for child in tree.children:
#       if child.name.startswith(str(val)):
#         tree = child
#   if (tree.name.endswith('1') and y == 1) or (tree.name.endswith('0') and y == 0):
#     return True
#   return False

# print(ten_fold_cross_validation(xd, y, index_to_feature))

# # problem 4.7
# def build_random_forest_subset(xd, y, itf):
#   trees = []
#   for iteration in range(5):
#     random_indices = []
#     for i in range(int(N * 0.2)):
#       r = random.randint(0,N)
#       if r not in random_indices:
#         random_indices.append(r)
#     random_x = []
#     random_y = []
#     for index in range(len(y)):
#       if index in random_indices:
#         continue
#       random_y.append(y[index])
#     for feature in range(len(xd)):
#       data = xd[feature]
#       random_feature = []
#       for index in range(len(data)):
#         if index in random_indices:
#           continue
#         random_feature.append(data[index])
#       random_x.append(random_feature)
#     tree = build_decision_tree(random_x, random_y, None, None, itf)
#     trees.append(tree)
#     # print_tree(tree)
#   return trees

# # problem 4.8
# def build_random_forest_feature(xd, y, itf):
#   trees = []
#   for i in range(len(xd)):
#     copy_x = copy.deepcopy(xd)
#     copy_x.remove(copy_x[i])
#     copy_itf = copy.deepcopy(itf)
#     copy_itf.remove(copy_itf[i])
#     tree = build_decision_tree(copy_x, y, None, None, copy_itf)
#     trees.append(tree)
#     # print_tree(tree)
#   return trees

# def ten_fold_cross_validation_rf(xd, y, itf):
#   correct = 0
#   for i in range(10):
#     start = i * N//10
#     end = (i+1) * (N//10)
#     trainingx = []
#     for i in range(len(xd)):
#       training_feature = []
#       for j in range(len(xd[i])):
#         if j < start or j > end:
#           training_feature.append(xd[i][j])
#       trainingx.append(training_feature)
#     trainingy = y[:start] + y[end + 1:]

#     trees = build_random_forest_feature(trainingx, trainingy, itf)
#     for index in range(i * N//10, (i+1) * (N//10)):
#       true_x = []
#       for i in range(len(xd)):
#         true_x.append(xd[i][index])
#       true_y = y[index]
#       corr_cnt = 0
#       for ti in range(len(trees)):
#         if predict(trees[ti], true_x, true_y, itf):
#           corr_cnt += 1
#       if corr_cnt >= len(trees) / 2:
#         correct += 1
#   return correct / N

# # build_random_forest_subset(xd, y, index_to_feature)
# # build_random_forest_feature(xd, y, index_to_feature)

# print(ten_fold_cross_validation_rf(xd, y, index_to_feature))
