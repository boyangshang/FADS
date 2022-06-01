"""
Unit tests for the FADS package.
"""

# Import package, test suite, and other packages as needed
import FADS
import pytest
import sys
import numpy as np
import warnings
import scipy.stats as ss

def test_FADS_imported():
    """Test whehter the FADS package can be imported or not."""
    assert "FADS" in sys.modules



#We use the same data as in the README.md file.
@pytest.fixture
def data4tests():
    
    ##Generate Synthetic data
    dimension = 2
    N = 10**4

    q = dimension
    mu1 = np.zeros(q, dtype = np.float64)
    mu2 = np.zeros(q, dtype = np.float64)
    for i in range(q):
        mu2[i] = 5.0 * (-1)**(i-1)

    sigma12 = 2.0
    sigma22 = 4.0
    alpha1 = 2.0
    alpha2 = 1.0
    D = np.eye(q, dtype = np.float64)
    a1 = np.ones(q, dtype = np.float64)
    a2 = np.zeros(q, dtype = np.float64)
    for i in range(q):
        a2[i] = 0.2*(i-1)*(-1)**(i-1)

    Sigma2 = sigma22 * D + alpha2 * np.matmul(a2, a2.T)
    Sigma1 = Sigma2

    data = np.zeros(shape = (N,dimension))


    idx = np.random.choice([True,False],N,p=[0.5,0.5])
    data[idx,:] = np.random.multivariate_normal(mu1, Sigma1, size=np.sum(idx))
    data[~idx,:] = np.random.multivariate_normal(mu2, Sigma2, size=N-np.sum(idx))
    
    return data

#test the diversity subsampling (DS) function
def test_DS(data4tests):
    
    #subsample size
    nmax = 2000
    
    #pass our generated synthetic data
    data = data4tests
    
    #perform diversity subsampling
    fastds = FADS.FADS(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_idx = fastds.DS(nmax)

    #the coordinates the selected subsample
    ds_sample = data[ds_idx,:]
    
    
#test the diveristy subsampling with replacement (DS-WR) function
def test_DS_WR(data4tests):
    
    #subsample size
    nmax = 2000
    
    #pass our generated synthetic data
    data = data4tests
    
    #perform diversity subsampling
    fastds = FADS.FADS(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ds_idx = fastds.DS_WR(nmax)

    #coordinates of the selected subsample
    sample = data[ds_idx,:]
    
#test the custom subsampling (CS) function
def test_CS(data4tests):
    
    #pass our generated synthetic data
    data = data4tests
    
    #target subsampling ratios
    desired_ratios = ss.norm(-2.5, 1).pdf(data[:,0])*ss.norm(2.5, 1).pdf(data[:,1])

    #subsampling using the DS_g function under the 
    #default hyper-paramter settings
    nmax = 2000
    reg_param = 50

    #custom subsampling
    fastds = FADS.FADS(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_g_idx = fastds.DS_g(nmax, target_pdf_list = desired_ratios, reg_param = reg_param)

    #coordinates of selected subsample
    mysubsample = data[ds_g_idx,:]
    
#test the parameter tuning function
def test_tune_params(data4tests):
    
    #pass our generated synthetic data
    data = data4tests
    
    fastds = FADS.FADS(data, tune_params = True)
    #user-specified parameter choices
    nfold = 3
    ncomponents = [2,15,30]
    max_iters = [10,100]
    inits = ['kmeans']
    fraction = 0.5

    #tune hyper-parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fastds.tune_params_CV(ncomponent_list = ncomponents,\
           max_iter_list = max_iters, init_list = inits,\
           nfold = nfold,fraction = fraction)

    #e.g. use DS to select a diverse subsample with size
    #     2000 using the tuned paramters
    n = 2000
    ds_idx = fastds.DS(n)


