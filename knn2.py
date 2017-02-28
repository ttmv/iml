#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats.mstats import mode


def count_miscalc_rate(trues, predicted, total):
  dsums = np.diag(trues+predicted)
  ones = dsums[dsums[:]==1]
  return (len(ones) / (total+0.0))


def count_modes(testlabels, slabels, k):
  ksums = np.sum(slabels[:,:k], axis=1)/(k+0.0)
  ksums = np.round(ksums)
  sumc = count_miscalc_rate(testlabels, ksums, len(testlabels))
  return sumc


def KNN(trainset, testset, trainlabels, testlabels, k_vals):
  knn_errors = []
  distmat = cdist(testset, trainset)
  slabels = trainlabels[np.argsort(distmat)]
  for k in k_vals:
    knn_errors.append(count_modes(testlabels, slabels, k))
  return knn_errors


def classify(x1, x2):
  bound3 = 5.91485594
  xsum = np.power(x1,2)+np.power(x2,2) 
  if xsum > bound3:
    return 1
  else:
    return 0


def bayes_error():
  cl, bpoints = count_points(10000)

  pred_class = np.zeros(len(cl))
  for i in range(len(cl)):
    pred_class[i] = classify(bpoints[0][i], bpoints[1][i])
  coms = cl+pred_class
  errs = coms[coms[:]==1]
  
  return np.sum(errs)/10000


def samples2(classes):
  total = len(classes)
  amount1 = np.sum(classes)
  x01 = np.random.normal(0, 1, total-amount1)
  x02 = np.random.normal(0, 1, total-amount1)
  x11 = np.random.normal(0, 4, amount1)
  x12 = np.random.normal(0, 4, amount1)
  
  all1 = np.zeros(total)
  all2 = np.zeros(total)
  all1[:total-amount1] = all1[:total-amount1]+ x01
  all1[total-amount1:] = all1[total-amount1:] + x11

  all2[:total-amount1] = all2[:total-amount1]+ x02
  all2[total-amount1:] = all2[total-amount1:] + x12
  return (all1, all2)


def count_points(total):
  classes = np.random.choice([0,1], total)
  amount = np.sum(classes)
  class2 = np.zeros(len(classes))
  class2[len(classes) - np.sum(classes):] = class2[len(classes) - np.sum(classes):] + 1
  return (class2, np.array(samples2(classes)))


def plot_set(point_set, all1, all2, total, amount1):
  plt.plot(all1[total-amount1:], all2[total-amount1:], 'bo', all1[:total-amount1], all2[:total-amount1], 'r^')
  plt.show()


def plot_fig(kvals, test_err, train_err, bayes_err):
  yticks = np.arange(0.0, 0.2, 0.015)
  plt.xticks(kvals, kvals)
  plt.yticks(yticks, yticks)
  plt.ylabel('error rate')
  plt.gca().invert_xaxis()
  plt.gca().xaxis.tick_top()

  plt.xlabel('k')
  plt.gca().xaxis.set_label_position('top')

  test, = plt.plot(kvals, test_err, 'r-o', label="test")
  train, = plt.plot(kvals, train_err, 'c-^', label="train")
  bayes, = plt.plot(kvals, np.zeros(len(kvals))+bayes_err, 'b', label="bayes")
  plt.legend(loc=3)
  plt.show()


tr_class, tr_set = count_points(500)
test_class, test_set = count_points(2000)
plot_set(tr_set, tr_set[0], tr_set[1], 500, np.sum(tr_class))
plot_set(test_set, test_set[0], test_set[1], 2000, np.sum(test_class))

trset = np.matrix.transpose(tr_set)
testset = np.matrix.transpose(test_set)
k_vals = [1,3,5,7,9,13,17,21,25,33,41,49,57]
test_errors = KNN(trset, testset, tr_class, test_class, k_vals)
print "test errors: ",test_errors
trainset_errors = KNN(trset, trset, tr_class, tr_class, k_vals)
print "train errors: ", trainset_errors
bayes_err = bayes_error()
print "bayes error: ", bayes_err
plot_fig(k_vals, test_errors, trainset_errors, bayes_err)


