import numpy as np
import mnist_load_show as mnist
import time


X, y = mnist.read_mnist_training_data()

datasize = len(y)/2
y2 = np.zeros(len(y))
tX = X[:datasize]
tY = y[:datasize]
testX = X[datasize:]
testY = y[datasize:]
classm = np.zeros((10, datasize))

#----------------------------------------------

def count_conf_matrix(trues, predicted):
    confm = np.zeros((10,10))
    for i in range(10):
      temp = predicted[trues[:] == i]
      for j in range(10):
        confm[i][j] = len(temp[temp[:] == j]) 
    return confm

def cm_to_str(cm):
    cmstr = ""
    for i in range(10):
      for j in range(10):
        cmstr = cmstr+'\t'+str(int(cm[i][j]))
      cmstr = cmstr+'\n'
    return cmstr

def print_cm(cm):
    print cm_to_str(cm)
    #print np.diag(cm)
    #print np.sum(np.diag(cm))
    print "error rate: ", (1-np.sum(np.diag(cm))/datasize)*100, "%"


#----------------------------------------------

def sign(cur_x, cur_w):
    if np.dot(cur_w, cur_x) > 0:
      return 1
    return -1

def reduce_ovsa_class():
    for i in range(10):
      classm[i,tY[:]==i] = 1
      classm[i,tY[:]!=i] = -1
    return classm

def perceptron(max_epoch, wsize, binY):
    w = np.zeros(wsize)
    pocket = (0, w)
    score = 0
    pocket2 = (0,w)

    for epoch in range(max_epoch):
      converged = True
      for i in range(datasize):
        if(binY[i] != 0):
          y_est = sign(w, tX[i])
          if y_est == binY[i]:
            score = score + 1
          else:
            if score > pocket[0]:
              pocket = (score, w)   
            w = w + binY[i] * tX[i]
            converged = False
            score = 0
      if converged:
        return w
      
      if pocket2[0] >= pocket[0]:
        return pocket2[1]
      pocket2 = pocket
      
    return pocket2[1]  


def train_one_vs_all(epoch):
    classm = reduce_ovsa_class()
    wsize = len(X[0])
    arrw = []
    for i in range(10):
      arrw.append(perceptron(epoch, wsize, classm[i]))
    return arrw


def create_ij(ii, jj):
  yi = np.zeros(len(tY))
  yi[tY[:]==ii] = yi[tY[:]==ii] + 1
  yi[tY[:]==jj] = yi[tY[:]==jj] - 1
  return yi


def reduce_class_avsa():
    yall = []
    for i in range(10):
      yis = []
      for j in range(10):
        yis.append(create_ij(i, j))
      yall.append(yis)
    
    return yall  
  

def create_avsa_vectors(binYs, epoch):
    W  = np.zeros((10,10,len(X[1])))
    for i in range(10):
      for j in range(i): 
        res = perceptron(epoch, len(X[0]), binYs[i][j])
        W[i][j] = res
        W[j][i] = -1*res
    return W


def classify_ovsa(W):
    rc = 0
    preds = np.zeros((10, datasize))
    for w in W:
      for i in range(datasize):
        preds[rc][i] = np.sum(w.T * testX[i])
      rc = rc+1
    
    return preds


def classf(W, x):
  a = np.zeros((10,10))
  for i in range(10):
    for j in range(10):
      a[i][j] = sign(x, W[i][j])
  return np.argmax(np.sum(a, axis=1))


def classify_avsa(W):
  preds = np.zeros(datasize)
  for i in range(datasize):
    preds[i] = classf(W, testX[i])
  return preds


def one_vs_all():
    """
    Implement the the multi label classifier using one_vs_all paradigm and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using the classifier
    """
    epoch = 30
    W = train_one_vs_all(epoch)
    predY = np.argmax(classify_ovsa(W), axis=0)
    one_vs_all_conf_matrix = count_conf_matrix(testY, predY)
    #print_cm(one_vs_all_conf_matrix)
    return one_vs_all_conf_matrix


def all_vs_all():
    """
    Implement the multi label classifier based on the all_vs_all paradigm and return the confusion matrix
    :return: the confusing matrix obtained regarding the result obtained using teh classifier
    """
    epoch = 30
    binYs = reduce_class_avsa()
    W = create_avsa_vectors(binYs, epoch)
    predY = classify_avsa(W)
    all_vs_all_conf_matrix = count_conf_matrix(testY, predY)
    #print_cm(all_vs_all_conf_matrix)
    
    return all_vs_all_conf_matrix


