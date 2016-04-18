#!/usr/bin/python

"""
@author: hexuan
"""

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import norm
from sklearn.metrics import accuracy_score

def read_data(trpath,tepath):
    '''
    read in the file as matrix
    '''
    trfin = open(trpath, 'r')
    tefin = open(tepath, 'r')
    X_tr = np.zeros((32561, 123), dtype=np.int)
    X_te = np.zeros((16281, 123), dtype=np.int)
    y_tr = np.ones((32561, ), dtype=np.int)
    y_te = np.ones((16281, ), dtype=np.int)
    r1 = 0
    r2 = 0
    for train in trfin:
        temp = train.strip().split()
        if temp[0] == '-1':
            y_tr[r1] = 0
        for i in temp[1:]:
            fid = i.split(':')
            X_tr[r1][int(fid[0])-1] = int(fid[1])
        r1 += 1
    for test in tefin:
        temp = test.strip().split()
        if temp[0] == '-1':
            y_te[r2] = 0
        for i in temp[1:]:
            fid = i.split(':')
            X_te[r2][int(fid[0])-1] = int(fid[1])
        r2 += 1
    trfin.close()
    tefin.close()
    return X_tr, X_te, y_tr, y_te

def sigmoid(x):
    y = float(1) / (1 + np.exp(-x))
    return y

def irls(W, X_tr, y_tr, lam):
    sigma = sigmoid(np.dot(X_tr, W)) #calculate sigmoid values of sigma
    R = np.zeros((sigma.shape[0], sigma.shape[0])) #the R N X N matrix
    for i in range(len(sigma)):
        R[i][i] = sigma[i] * (sigma[i] - 1)
    dL = np.dot(X_tr.T, y_tr - sigma) - lam * W # dL(w)
    H = np.dot(np.dot(X_tr.T, R), X_tr) - lam * np.eye(X_tr.shape[1]) #The Hessian matrix
    Hv = pinv(H)
    Wn = W - np.dot(Hv, dL) #derive new W
    return Wn

def likelihood(W, X_tr, y_tr):
    '''
    calculate the log likelihood over all the samples.
    '''
    X = np.dot(X_tr, W)
    sigma = sigmoid(X)
    p = np.sum(np.log((sigma))) / 100000.0 #rescale the likelihood to (-1, 0) for convinient to plot convergence curve.
    return p

def training(W, X_tr, y_tr, lam):
    '''
    train the model according to IRLS.
    '''
    p_old = likelihood(W, X_tr, y_tr)
    W_new = irls(W, X_tr, y_tr, lam)
    p_new = likelihood(W_new, X_tr, y_tr)
    ites = 1
    print 'Iterations:',ites,'\t','likelihood:',p_new
    print '=' * 50
    while p_new != p_old:
        W = W_new.copy()
        p_old = likelihood(W, X_tr, y_tr)
        W_new = irls(W, X_tr, y_tr, lam)
        p_new = likelihood(W_new, X_tr, y_tr)
        ites += 1
        print 'Iterations:',ites,'\t','likelihood:',p_new
        print '=' * 50
    return W_new, ites
    
def prediction(W, X_te, y_te):
    y = np.dot(X_te, W)
    y_pred = np.array([int(sigmoid(i) + 0.5) for i in y])
    s = accuracy_score(y_te, y_pred)
    return s
    
if __name__ == '__main__':
    trpath, tepath = './a9a', './a9a.t'
    X_tr, X_te, y_tr, y_te = read_data(trpath, tepath)
    np.random.seed(0)
    W = np.random.random(X_tr.shape[1]) * 0.1 #for avoiding the flow-out problem in sigmoid fuction
    W_trd, iterations = training(W, X_tr, y_tr, 28.5)
    #W_trd, iterations = training(W, X_tr, y_tr, 0)
    score1 = prediction(W_trd, X_te, y_te)
    score2 = prediction(W_trd, X_tr, y_tr)
    W_norm = norm(W_trd)
    print 'Accuracy:', 'test:',score1,'\t','train:',score2
    print 'Norm is:',W_norm
    print 'Iterations:', iterations
