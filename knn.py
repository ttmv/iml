import numpy as np
import mnist_load_show as mnist


from scipy.spatial.distance import cdist


X, y = mnist.read_mnist_training_data(5000)
trainset = X[:2500]
trainlabels = y[:2500]
testset = X[2500:]
testlabels = y[2500:]


def count_conf_matrix(trues, predicted):
    confm = np.zeros((10,10))

    for i in range(10):
        temp = predicted[trues[:] == i]
        confm[i][i] = len(temp[temp[:] == i]) #amount of correct values
        for j in range(10):
            confm[i][j] = len(temp[temp[:] == j]) #the rest
    print "confusion matrix:"        
    print confm        
    return confm


def KNN():
    """
    :return: the confusion matrix regarding the result obtained using knn method
    """
    distmat = cdist(trainset, testset)
    vals = np.zeros(len(testset))
    for i in range(len(testset)):
        vals[i] = trainlabels[np.argmin(distmat[:,i])]
    knn_conf_matrix = count_conf_matrix(testlabels, vals)

    return knn_conf_matrix


def count_prototype(classnro):
    czm = trainset[trainlabels[:] == classnro]
    return np.mean(czm, axis=0)	

def create_prototypes():
    protos = []
    for i in range(10):
        protos.append(count_prototype(i))

    return np.array(protos)


def simple_EC_classifier():
    """
    :return: the confusing matrix obtained regarding the result obtained using simple Euclidean distance method
    """
    protos = create_prototypes()
    distmat = cdist(protos, testset)
    vals = np.zeros(len(testset))
    for i in range(len(testset)):
        vals[i] = np.argmin(distmat[:,i])
    simple_EC_conf_martix = count_conf_matrix(testlabels, vals)

    return simple_EC_conf_martix


