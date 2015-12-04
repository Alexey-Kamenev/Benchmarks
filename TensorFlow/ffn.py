import tensorflow as tf
import numpy as np

featureDim = 512
labelDim = 10000
hiddenLayDim = 2048
numMinibatches = 150

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('logDevicePlacement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('noInputFeed', False,
                            """Whether to not feed new features/labels data for each minibatch.""")

data = np.loadtxt('../data.txt')
features = data[:,1:]
labels = data[:,0]

# Get random parameters initialized with a iniform distribution between -0.5 and 0.5
def getParameters(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5))

def sigmoidDNNLayer(layerIdx, input, inputDim, outputDim):
    W = getParameters("W" + str(layerIdx), [inputDim, outputDim])
    B = getParameters("B" + str(layerIdx), [outputDim])
    return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))

def getFakeMinibatch(minibatchSize):
    #feat = np.random.randn(minibatchSize, featureDim)
    #lab = np.zeros((minibatchSize, labelDim))
    #for row in lab:
    #	row[np.random.randint(0, labelDim)] = 1
    feat = features[:minibatchSize]
    l = labels[:minibatchSize]
    lab = np.zeros((minibatchSize, labelDim))
    for i in range(lab.shape[0]):
        lab[i][l[i]] = 1
    return feat, lab
    #fakeFeatures = [[0.0 for _ in xrange(featureDim)] for _ in xrange(minibatchSize)]
    #fakeLabels = [[0.0 for _ in xrange(labelDim)] for _ in xrange(minibatchSize)]
    #for sampleIdx in xrange(minibatchSize):
    #    fakeLabels[sampleIdx][np.random.randint(0, labelDim - 1)] = 1.0
    #    for featureIdx in xrange(featureDim):
    #        fakeFeatures[sampleIdx][featureIdx] = np.random.randn()
    #
    #return fakeFeatures, fakeLabels


def getLossAndAccuracyForSubBatch(features, labels):

    HL0 = sigmoidDNNLayer(0, features, featureDim, hiddenLayDim)
    HL1 = sigmoidDNNLayer(1, HL0, hiddenLayDim, hiddenLayDim)
    HL2 = sigmoidDNNLayer(2, HL1, hiddenLayDim, hiddenLayDim)
    HL3 = sigmoidDNNLayer(3, HL2, hiddenLayDim, hiddenLayDim)

    outputLayerW = getParameters("W5", [hiddenLayDim, labelDim])
    outputLayerB = getParameters("B5", [labelDim])
    outputLayer = tf.nn.softmax(tf.nn.xw_plus_b(HL3, outputLayerW, outputLayerB))

    crossEntropy = -tf.reduce_mean(labels * tf.log(outputLayer))
    predictionCorrectness = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(predictionCorrectness, "float"))

    return crossEntropy, accuracy

def printTrainingStats(numGPUs, minibatchSize, perMinibatchTime):
    meanTimePerMinibatch = np.mean(perMinibatchTime)
    medianTimePerMinibatch = np.median(perMinibatchTime)
    minTimePerMinibatch = np.min(perMinibatchTime)

    def samplesPerSec(minibatchSize, processingTime):
        return minibatchSize/processingTime

    print('*****************************Training on %d GPUs***************************************' % numGPUs)
    print('MinibatchSize=%d, NumMinibatches=%d.' % (minibatchSize, numMinibatches))
    print('Training speed (samples/sec): Average=%d, Median=%d, Max=%d' % (samplesPerSec(minibatchSize, meanTimePerMinibatch),
                                                                           samplesPerSec(minibatchSize, medianTimePerMinibatch),
                                                                           samplesPerSec(minibatchSize, minTimePerMinibatch)))
    print('*************************************************************************************')



