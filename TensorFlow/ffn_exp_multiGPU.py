# A feed-forward DNN with 5 hidden layers using sigmoid activations.
# Uses dataparallel SGD with multiple GPUs

import time
import tensorflow as tf
import ffn

from ffn import *

tf.app.flags.DEFINE_integer('numGPUs', 4,
                            """How many GPUs to use.""")

subMinibatchSize = 512
minibatchSize = FLAGS.numGPUs * subMinibatchSize

def aggregateGradients(subMinibatchGradients):
  aggGrads = []
  for gradAndVars in zip(*subMinibatchGradients):
    # Note that each gradAndVars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in gradAndVars:
      # Add 0 dimension to the gradients to represent the replica.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'replica' dimension which we will sum over below.
      grads.append(expanded_g)

    # Sum over the 'replica' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_sum(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across replicas. So .. we will just return the first replica's pointer to
    # the Variable.
    v = gradAndVars[0][1]
    gradAndVar = (grad, v)
    aggGrads.append(gradAndVar)
  return aggGrads

if (FLAGS.noInputFeed):
  features, labels = getFakeMinibatch(subMinibatchSize)
else:
# HACK: Using the same subMinibatch across all GPUs
  features = tf.placeholder("float", [None, featureDim])
  labels = tf.placeholder("float", [None, labelDim])

optimizer = tf.train.GradientDescentOptimizer(0.01)

# Calculate the gradients for each subBatch on a different GPU
subMinibatchGradients = []
subMinibatchAccuracies = []
for i in xrange(FLAGS.numGPUs):
  with tf.device('/gpu:%d' % i):
    with tf.name_scope('%s_%d' % ("replica", i)) as scope:
      # Calculate the loss for one subBatch. This function
      # constructs the entire model but shares the variables across
      # all replicas.
      loss, accuracy = getLossAndAccuracyForSubBatch(features, labels)

      # Reuse variables for the next replica.
      tf.get_variable_scope().reuse_variables()

      # Calculate the gradients for this subBatch on this GPU
      grads = optimizer.compute_gradients(loss)

      # Keep track of the gradients across all replicas.
      subMinibatchGradients.append(grads)
      subMinibatchAccuracies.append(accuracy)

# We must calculate the sum of each gradient. Note that this is the
# synchronization point across all towers.
grads = aggregateGradients(subMinibatchGradients)
accuracy = tf.reduce_sum(tf.pack(subMinibatchAccuracies))

# Apply the gradients to adjust the shared variables.
applyGradientOp = optimizer.apply_gradients(grads)

# Start running operations on the Graph. allow_soft_placement must be set to
# True to build replicas on GPU, as some of the ops do not have GPU implementations.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.logDevicePlacement))
init = tf.initialize_all_variables()
sess.run(init)

# Start the queue runners.
tf.train.start_queue_runners(sess=sess)

perMinibatchTime = []
for step in xrange(numMinibatches):
  if (FLAGS.noInputFeed == False):
    subMinibatchFeatures, subMinibatchLabels = getFakeMinibatch(subMinibatchSize)

  startTime = time.time()
  if (FLAGS.noInputFeed):
    sess.run([applyGradientOp, accuracy])
  else:
    sess.run([applyGradientOp, accuracy], feed_dict={features: subMinibatchFeatures, labels: subMinibatchLabels})

  currMinibatchDuration = time.time() - startTime
  perMinibatchTime.append(currMinibatchDuration)

printTrainingStats(FLAGS.numGPUs, minibatchSize, perMinibatchTime)

