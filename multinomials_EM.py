#!/usr/bin/python

import numpy as np
#from sys import argv

def get_data(lpath, vpath):
    lfin = open(lpath, 'r') #nips.libsvm
    vfin = open(vpath, 'r') #nips.vocab
    wf = {} #words_id:words_frequency
    wv = {} #words_id:words
    Tdw = [] #word occurence matrix
    for row in vfin:
        row = row.strip().split()
        wf[row[0]] = int(row[2])
        wv[row[0]] = row[1]
    t = []
    for i in range(len(wf)):
        t.append(0)
    for doc in lfin:
        doc = doc.strip().split()
        temp = t[:]
        for word in doc[1:]:
            word = word.split(':')
            temp[int(word[0])] = int(word[1])
        Tdw.append(temp[:])
    lfin.close()
    vfin.close()
    Tdw = np.array(Tdw)
    return wf, wv, Tdw

def initiation(wf, K):
    W = len(wf)
    np.random.seed(0)
    pik = np.random.dirichlet(np.ones(K), 1)[0]
    mut = np.random.dirichlet(np.ones(W), K)
    mu = mut.T
    return pik, mu

def E_step(Tdw, pik, mu):
    pik = np.log(np.add(pik, 1e-100))
    mu = np.log(np.add(mu, 1e-100))
    val = np.dot(Tdw, mu) + pik
    t = val.max(axis=1)
    val = np.exp((val.T - t).T)
    Z = (val.T / val.sum(axis=1).T).T
    return Z

def M_step(Z, Tdw):
    npik = Z.sum(axis=0) / len(Tdw)
    dsum = np.dot(Tdw.T, Z)
    total = dsum.sum(axis=0)
    nmu = dsum / total
    return npik, nmu

def main(lpath, vpath, K):
    K = int(K)
    wf, wv, Tdw = get_data(lpath, vpath)
    pik, mu = initiation(wf, K)
    threshold = 1e-20
    while 1:
        Z = E_step(Tdw, pik, mu)
        pik_new, mu_new = M_step(Z, Tdw)
        dp = np.sqrt(np.sum((pik_new - pik)**2))
        dm = np.sqrt(np.sum((mu_new - mu)**2))
        print ('=' * 60)
        print ('dp:',dp)
        print ('dm:',dm)
        if dp < threshold and dm < threshold:
            print ('=' * 60)
            print ('K:', K)
            print ('topic words for each K:')
            for w in mu_new.T.tolist():
                idx = str(w.index(max(w)))
                print (wv[idx])
            break
        else:
            pik, mu = pik_new.copy(), mu_new.copy()

if __name__ == '__main__':
    #lpath, vpath, K = argv[1], argv[2], argv[3]
    lpath, vpath, K = './data/nips/nips.libsvm', './data/nips/nips.vocab', 30 
    main(lpath, vpath, K)        