import numpy as np
import os
import gzip
from struct import Struct

# path and files
path = './data/mnist/'
trainingSetFile = 'train-images-idx3-ubyte.gz'
trainingLabelFile = 'train-labels-idx1-ubyte.gz'
TestSetFile = 't10k-images-idx3-ubyte.gz'
TestLabelFile = 't10k-labels-idx1-ubyte.gz'
# records struct
headerRecord = Struct('>iiii')
# 

def readImage(format, f):
    imageRecord = Struct(format)
    chunks = iter(lambda: f.read(imageRecord.size), b'')
    return (imageRecord.unpack(chunk) for chunk in chunks)

with gzip.open(os.path.join(path, trainingSetFile), 'rb') as f:
    magic, m, width, height = headerRecord.unpack(f.read(headerRecord.size))
    trainingSet = np.zeros((m, width * height))
    for index, img in enumerate(readImage('>'+'B'*(width * height), f)):
        trainingSet[index,] = img
    trainingSet = trainingSet.reshape(m, width, height)

print(trainingSet[0,])