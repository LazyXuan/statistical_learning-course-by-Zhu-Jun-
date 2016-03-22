# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:48:57 2016

@author: hexuan
"""

import gzip
import cPickle
import numpy as np
from numpy.linalg import svd
from PIL import Image
from utils import tile_raster_images

def get_data(dpath):
    '''
    Read in MNIST.pkl.gz data 
    '''
    fin = gzip.open(dpath, 'rb')
    train_set, test_set, cv_set = cPickle.load(fin)
    fin.close()    
    return train_set[0][:10000]#select first 10000 samples

def preprocess(data):
    '''
    feature scaling and normalization.
    '''
    m = len(data)
    mu = data.sum(axis=0) / m
    ndata = data - mu / 255.0
    return ndata

def pca(ndata):
    '''
    Select first 1, 5, 20, 100 components respectively
    '''
    Sigma = ndata.dot(ndata.T) / len(ndata)
    U, S, v = svd(Sigma)
    Urs = [U[:,:1],U[:,:5],U[:,:20],U[:,:100]]
    z = [Urs[i].T.dot(ndata) for i in range(4)]
    Ua = [Urs[j].dot(z[j]) for j in range(4)]
    return z, Ua

def plot(data, z, Ua):
    '''
    Display the images of original data, reduced data and reconstruct data.
    '''
    image = Image.fromarray(
        tile_raster_images(
            X=data[:100],
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )
    image.save('original.png')
    for i in range(4):
        zimage = Image.fromarray(
            tile_raster_images(
                X=z[i][:100],
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        zimage.save('reduced_k%i.png' % i)
        uimage = Image.fromarray(
            tile_raster_images(
                X=Ua[i][:100],
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        uimage.save('reconstructed_k%i.png' % i)

if __name__ == '__main__':
    dpath = './data/mnist.pkl.gz'
    data = get_data(dpath)
    ndata = preprocess(data)
    z, Ua = pca(ndata)
    z, Ua = pca(data)
    plot(data, z, Ua)
