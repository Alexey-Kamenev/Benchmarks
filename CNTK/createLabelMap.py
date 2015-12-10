import numpy as np
dim = 1000
a = range(0, dim)
np.savetxt('./labelmap.txt', np.reshape(a, (dim, 1)), fmt='%d')
