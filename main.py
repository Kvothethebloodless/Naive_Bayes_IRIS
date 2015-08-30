import numpy as np
import matplotlib.pyplot as plt #from scipy.stats
#import multivariate_normal
import parse_iris as pi
import math as math
#3 important functions.

# p(c_i/X) = (p(X/c_i)*p(c_i))/p(X)
# Posterior = Likelihood * Prior/Scaling factor


#inference_likelihood /Generates the multivariate gaussian describing likelihood function of each class
#inference_prior /Calculates Prior of each class
#decision_posterior /Calculates Posterior probability of test sample and decides label assignment


def norm_pdf_multivariate(x, param):
    
    [size] = np.shape(x)
    pi = 3.14;
    [mu,sigma] = param;
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu);
        inv = np.linalg.inv(sigma);       
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the inputs don't match")
    
    
def inference_prior(traindata):
    C_is = np.array([0,1,2])
    freq_Cis = np.array([np.sum(traindata[:,4]==C_i)  for C_i in C_is])
    prior_Cis =  freq_Cis/(np.sum(freq_Cis))
    return np.array(prior_Cis)
    

def decision_posterior(priors,Ci_params,datapoints,n_classes,n_datapoints):
    #Ci_params is a tuple with (class_labels,mu_array,var_matrix) as elements
    # Given n dimension data with I classes - size of Ci_params is 3 by 
    #given data point, calculate the posterior probabilities
    #and put it back with [Ci,posterior]
    #Return Ci with maximum posterior probability.
    (class_labels,mu_array,var_matrix) = Ci_params;
    posterior_prob = np.zeros((n_classes,2));
    posterior_prob[:,0] = class_labels; #Writing classes
    decided_class=np.arange(n_datapoints);
    for j in range(n_datapoints):
        datapoint =datapoints[j];
        for i in range(n_classes):
            posterior_prob[i,1] = norm_pdf_multivariate(datapoint,(mu_array[i],var_matrix[i])) * priors[i] ;
        max_post_prob_loc = np.argmax(posterior_prob[:,1]);                                  
        max_post_class = posterior_prob[:,0][max_post_prob_loc]
        decided_class[j] = max_post_class;
    return decided_class

def inference_likelihood(traindata,class_labels,n_classes,n_features): #Naive Bayes, Class Conditional Independence assumption + MLE estimation
    #return a matrix with parameters for each class.
    #return a tuple with 3 nparrays - class_labels,mu_array,var_matrix
    
    all_class_mu = np.zeros((n_classes,n_features));
    all_class_var = np.zeros((n_classes,n_features,n_features));
    
    for i in range(n_classes):
        data_ci = traindata[np.where((traindata[:,4]==class_labels[i]))];
        data_ci = data_ci[:,0:4];
        mu_array = np.mean(data_ci,axis=0); #Maximum Likelihood
        var_array = np.var(data_ci,axis=0); #Maximum Likelihood
        var_matrix = np.diag(var_array);
        all_class_mu[i] = mu_array;
        all_class_var[i] = var_matrix;
    all_class_params = (class_labels,all_class_mu,all_class_var) ;
    return all_class_params

def calc_error(assigned_labels,orig_labels):
    correct = np.sum(assigned_labels == orig_labels);
    total = np.size(assigned_labels);
    accuracy = correct / total;
    return accuracy*100;
    


if __name__ == "__main__":
    [traindata,testdata] = pi.get_data();
    n_classes = 3;
    n_features = 4;
    n_datapoints = 30;
    testdata_labels = testdata[:,4];
    testdata = testdata;
    all_class_labels = np.arange(3);

    priors = inference_prior(traindata);
    
    class_params = inference_likelihood(traindata,all_class_labels,n_classes,n_features);
    assigned_labels = decision_posterior(priors,class_params,testdata[:,0:4],n_classes,n_datapoints);


    print(calc_error(assigned_labels,testdata_labels))
