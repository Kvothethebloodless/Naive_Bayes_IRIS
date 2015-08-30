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


def norm_pdf_multivariate(x, param):
    size = len(x)
    [mu,sigma] = param;
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
    
    
def inference_prior(iris_traindata):
    C_is = np.array([0,1,2])
    freq_Cis = np.array([np.sum(iris_traindata[:,4]==C_i)  for C_i in C_is])
    prior_Cis =  freq_Cis/(np.sum(freq_Cis))
    return np.array([C_is,prior_Cis])
    

def decision_posterior(Ci_params,datapoint,n):
    #Ci_params is in accordance with [Ci,[mu_matrix,var_matrix]] given data point, calculate the posterior probabilities
    #and put it back with [Ci,posterior]
    #Return Ci with maximum posterior probability.
    posterior_prob = np.zeros((3,2));
    posterior_prob[:,0] = Ci_params[:,0]; #Writing classes
    posterior_prob[:,1] = norm_pdf_multivariate(datapoint,Ci_params[:,1]);
    max_post_prob_loc = np.argmax(posterior_prob[1]);                                  ;
    max_post_class = posterior_prob[:,0][max_post_prob_loc]
    

    
    
 










if __name__=='main':
    [iris_traindata,iris_testdata] = pi.get_data();
    no_classes = 3




