#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import mnist_load_show as mnist
import time

#datasize = 30000
X, y = mnist.read_mnist_training_data()
datasize = len(y)/2
y2 = np.zeros(len(y))
tX = X[:datasize]
tY = y[:datasize]
testX = X[datasize:]
testY = y[datasize:]
classm = np.zeros((10, datasize))



def count_conf_matrix(trues, predicted):
  confm = np.zeros((10,10))
  for i in range(10):
    temp = predicted[trues[:] == i]
    for j in range(10):
      confm[i][j] = len(temp[temp[:] == j]) 
  return confm


def signcount(cur_x, cur_w):
  s = np.dot(cur_w, cur_x)
  if s > 0:
    return 1
  else: 
    return -1

def reduce_class():
  for i in range(10):
    classm[i,tY[:]==i] = 1
    classm[i,tY[:]!=i] = -1
      
def perceptron(max_epoch, wsize, origclass):
  w = np.zeros(wsize)
  pocket = (0, w)
  score = 0

  for epoch in range(max_epoch):
    converged = True
    for i in range(datasize):
      y_est = signcount(w, tX[i])
      if y_est == classm[origclass][i]:
        score = score + 1
      else:
        if score > pocket[0]:
          pocket = (score, w)   
        w = w + classm[origclass][i] * tX[i]
        converged = False
        
    if converged:
      return w
   
  return pocket[1]
  

def train_one_vs_all(epoch):
  reduce_class()
  wsize = len(X[0])
  arrw = []
  for i in range(10):
    ltime = time.time()
    print "class ", i
    arrw.append(perceptron(epoch, wsize, i))
    print i, "done ", time.time() - ltime 
  return arrw


def classif2(curW):
  predy2 = np.zeros(datasize)
  for i in range(datasize):
    res = curW.T * testX[i]
    predy2[i] = np.sum(res)
  return predy2

def classify_one(curW):
  classes = []
  for x in testX:
    res = signcount(x, curW)
    classes.append(res)

  return classes

stime = time.time()
print "start 1 vs all"
W = train_one_vs_all(5)
print "done", time.time() - stime


def multiclasses(preds, real):
  predY = np.zeros(len(datasize))
  

def set_testclass(tc):
  yres0 = np.zeros(len(testY))
  yres0[testY[:]==tc] = 1
  yres0[testY[:]!=tc] = -1
  return yres0

preds = np.zeros((10, datasize))
realc = np.zeros((10, datasize))

def classify_test(W):
  rc = 0
  for w in W:
    real = set_testclass(rc)
    realc[rc] = real
    preds[rc] = classif2(w)
    rc = rc+1



def print_cm(cm):
  cmstr = ""
  for i in range(10):
    for j in range(10):
      cmstr = cmstr+'\t'+str(int(cm[i][j]))
    cmstr = cmstr+'\n'
  print cmstr
      
classify_test(W)
testclass = np.argmax(preds, axis=0)
cm = count_conf_matrix(testY, testclass)
print_cm(cm)
print np.diag(cm)
print np.sum(np.diag(cm))
print (1-np.sum(np.diag(cm))/datasize)*100
