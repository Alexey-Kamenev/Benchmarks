import numpy as np
a = range(0, 10000)
np.savetxt('./labelmap.txt', np.reshape(a, (10000, 1)), fmt='%d')
