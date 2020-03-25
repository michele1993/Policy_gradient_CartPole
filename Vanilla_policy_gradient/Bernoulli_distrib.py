import numpy as np

def Bern_sample(mean):

    r_n = np.random.rand()

    if r_n <= mean:

        return 1

    else:

        return 0