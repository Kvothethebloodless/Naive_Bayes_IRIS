import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import multivariate_normal
import parse_iris as pi

#3 important functions.

# p(c_i/X) = (p(X/c_i)*p(c_i))/p(X)
# Posterior = Likelihood * Prior/Scaling factor


#inference_likelihood /Generates the multivariate gaussian describing likelihood function of each class
#inference_prior /Calculates Prior of each class
#decision_posterior /Calculates Posterior probability of test sample and decides label assignment



def inference_prior(iris_traindata):
    C_is = np.array([0,1,2])
    freq_Cis = np.array([np.sum(iris_traindata[:,4]==C_i)  for C_i in C_is])
    prior_Cis =  freq_Cis/(np.sum(freq_Cis))
    return np.array([C_is,prior_Cis])
    















if __name__=='main':
    [iris_traindata,iris_testdata] = pi.get_data();
    




