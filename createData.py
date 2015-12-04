import numpy as np

featDim = 512
labDim = 10000
totalCount = 8 * 1024

def createFakeData(count):
    features = np.random.randn(count, featDim)
    labels = np.random.randint(0, labDim, size=(count, 1))
    return features, labels

f, l = createFakeData(totalCount)

np.savetxt(r'./data.txt', np.hstack((l, f)), fmt='%d' + ' %f4' * featDim)

