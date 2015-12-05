import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import theano
import time
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

nruns = 5
bsize = 8192
isize = 512
hsize = 2048
osize = 10000

#fake data
X = np.random.rand(bsize, isize).astype(np.float32)
y = np.zeros((bsize, osize), dtype=np.bool)
ind = np.random.randint(0,osize,bsize)
for i in range(bsize):
    y[i,ind[i]] = True

#model definition
model = Sequential()
model.add(Dense(hsize, input_dim=isize))
model.add(Activation('sigmoid')) #hidden layer 1
model.add(Dense(hsize))
model.add(Activation('sigmoid')) #hidden layer 2
model.add(Dense(hsize))
model.add(Activation('sigmoid')) #hidden layer 3
model.add(Dense(hsize))
model.add(Activation('sigmoid')) #hidden layer 4
model.add(Dense(osize))
model.add(Activation('softmax')) #output layer
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

#start training and measuring
start = time.time()
for i in range(nruns):
    model.train_on_batch(X, y)
end = time.time()
print('1 GPU: {0} samples per sec'.format(nruns * bsize / (end-start)))

