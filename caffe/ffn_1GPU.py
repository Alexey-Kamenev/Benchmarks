import numpy as np
import caffe as c
import time

mbSize = 512
count = mbSize * 16

feat = np.random.randn(count, 1, 1, 512).astype(np.float32)
lab = np.random.randint(0, 10000, size=(count, 1, 1, 1)).astype(np.float32)

def createSolver(solverFile):
    c.set_mode_gpu()
    solver = c.SGDSolver(solverFile)
    solver.net.set_input_arrays(feat, lab)
    return solver

def samplesPerSec(minibatchSize, processingTime):
    return minibatchSize / processingTime

def runBenchmark(solver, iter):
    startTime = time.time()
    solver.step(iter)
    stepTime = time.time() - startTime
    print "Samples per sec = %d." %  samplesPerSec(mbSize * iter, stepTime)

s = createSolver('./ffn_solver_md.prototxt')
#runBenchmark(s, 10)

