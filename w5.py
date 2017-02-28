#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_data():
  xvals = np.random.uniform(-3, 3, 30)
  n = np.random.normal(0, 0.4, 30)
  yvals = 2 + xvals - 0.5*xvals**2 + n
  return (xvals, yvals)


def create_xmat(exp, xvals):
  pows = np.matrix(np.arange(11))
  xmat = (xvals ** pows.T).T
  return xmat


def create_matrix(xmat, exp, Y):
  X = xmat[:,:exp]
  W = (X.T * X).I * X.T * Y 
  return (X, W, Y)


def c_of_d(y, predY):
  ydash = np.sum(y) / 30
  difs = y-predY
  ys = y-ydash
  divys = np.array(ys.T[0])[0]**2
  errors = np.array(difs.T[0])[0]**2
  return 1-(np.sum(errors) / np.sum(divys))


def plot_polynoms(xvals, yvals, predY, k, cod, axarr, s1, s2):
  axarr[s1][s2].set_title("K: "+str(k))
  axarr[s1][s2].plot(xvals, yvals, 'bo', np.sort(xvals), predY[np.argsort(xvals)], 'c-')


def plot_errors(errors):
  kdots = np.arange(11)
  plt.plot(kdots, errors)
  plt.show()


def subindexes(s1, s2):
  if s2 == 2:
    s2 = 0
    if s1 == 3:
      s1 = 0
    else:
      s1 = s1+1
  else:
    s2 = s2+1
  return (s1,s2)


def count_predY(fullmat, k, yvec):
  X, W, Y = create_matrix(fullmat, k, yvec) 
  xwt = np.array((X*W).T[0])[0]
  cod = c_of_d(Y, X*W)
  return (W, xwt, cod)


def inits(K):
  s1 = 0
  s2 = 1
  f, axarr = plt.subplots(4,3)
  axarr[0][0].set_title("Data")
  axarr[0][0].plot(xvals, yvals, 'bo')  
  for k in range(K):
    W, xwt, cod = count_predY(Xm11, k, y)
    plot_polynoms(xvals, yvals, xwt, k, cod, axarr, s1, s2)
    s1, s2 = subindexes(s1,s2)
  plt.tight_layout()
  plt.show()


def slice_mat(j_start, j_end, k):
  xj = Xm11[j_start:j_end,:k]
  yj = y[j_start:j_end]
  x_rest = np.concatenate([Xm11[:j_start,:k], Xm11[j_end:,:k]])  
  y_rest = np.concatenate([y[:j_start], y[j_end:]])
  return (yj, xj, y_rest, x_rest)


def count_err(yj, xj, y_rest, x_rest, W):
  diff = np.array((xj*W - yj).T[0])[0]**2
  return np.sum(diff)


def cross_val(K):
  errors = []
  print "K, errors:"    
  for k in range(K):
    j_start = 0
    j_end = 3
    error = 0

    while(j_end < len(xvals)):
      yj, xj, yrest, xrest = slice_mat(j_start, j_end, k)
      W = (xrest.T * xrest).I * xrest.T * yrest
      error = error + count_err(yj, xj, yrest, xrest, W)
      j_start = j_start + 3
      j_end = j_end + 3
      
    errors.append(error)
    print k, error
  plot_errors(errors)


xvals, yvals = create_data()
Kmax = 11
Xm11 = create_xmat(Kmax, xvals)
y = np.matrix(yvals).T

inits(Kmax)
cross_val(Kmax)


