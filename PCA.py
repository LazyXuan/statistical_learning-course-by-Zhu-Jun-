#!/usr/bin/python

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
    read in original data
    '''
    fin = gzip.open(dpath, 'rb')
    train_set, test_set, cv_set = cPickle.load(fin)
    fin.close()
    return train_set[0]

def preprocess(data):
    '''
    zero-centered the data
    '''
    ndata = np.copy(data)
    ndata -= np.mean(ndata, axis=0)
    return ndata

def pca(ndata):
    Sigma = np.dot(ndata.T, ndata) / ndata.shape[0]
    U, S, V = svd(Sigma)
    Urs = [U[:,:1].T,U[:,:5].T,U[:,:20].T,U[:,:100].T] # select first 1, 5, 20, 100 components.
    z = [np.dot(ndata, Urs[i].T) for i in range(4)]
    Ua = [np.dot(z[j], Urs[j]) for j in range(4)]
    dwhite = np.dot(ndata, U) / np.sqrt(S + 1e-10) #Whitening the data
    return Urs, z, Ua, dwhite

def plot(data, Urs, Ua, dwhite):
    '''
    plot the pictures results.
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

    image = Image.fromarray(
        tile_raster_images(
            X=dwhite[:100],
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )
    image.save('whitened.png')

    for i in range(4):
        zimage = Image.fromarray(
            tile_raster_images(
                X=Urs[i][:100],
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
    Urs, z, Ua, dwhite = pca(ndata)
    plot(data, Urs, Ua, dwhite)
