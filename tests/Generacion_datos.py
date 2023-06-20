import numpy as np
# Generation data definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(n_obs=1000, n_samples=100, n_important=10, scale=1, seed=None):
    """
    Parameters 
    -----------
    n_obs: Number of observations
    n_samples: Number of variables
    n_important: Number of truly important variables
    scale: Scale parameter in the error distribution
    seed: Seed for the random number generator. For reproducibility purposes

    Return
    -----------
    X: matrix of predictive variables
    y: vector of response
    beta: vector of coefficients for the variables
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(size=(n_obs, n_samples))
    beta = np.concatenate((np.random.uniform(low=1, high=10, size=n_important), np.zeros(n_samples-n_important))).reshape((-1,1))
    error = np.random.normal(size=(n_obs, 1), scale=scale)
    y = sigmoid(X@beta + error)
    y = np.array(y>0.5, dtype='int')
    return X, y, beta