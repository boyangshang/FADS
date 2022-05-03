
This mannual provides examples using the FAst Diversity Subsampling (**FADS**) package developed by [[1]](#1) to select diverse subsamples (**DS**) and custom subsamples (**CS**) from a data set in Python. As discussed in [[1]](#1), a diverse subsample is a subset of a data set that is spread out over the region occupied by the (usually unknown) data distribution. A custom subsample is selected using the user-defined target sampling ratios, with or without consideration of the diversity property.


This article is organized in the following way. [Installation](#installation) illustrates how to install this package. [Diversity Subsampling](#diversity-subsampling) shows how to use **FADS** to select a diverse subsample from a data set without or with replacement. [Custom Subsampling](#custom-subsampling) provides one example to use **FADS** to select a custom subsample. [Hyper-Parameter Tuning](#hyper-parameter-tuning) shows how to use the built-in function in **FADS** to further tune the hyper-parameters.

The experimental setting for this article is as follows. For easy visualization, throughout this article, we use a synthetic Multivariate Gaussian Mixture (MGM) data in 2D. Details about generating this data set are provided in [Appendix A](#appendix-A:-synthetic-data-used-in-this-article). The default hyper-parameter setting will be used for all examples in this article unless otherwise specified.

# Installation 
The **FADS** package is available on pip and can be installed using the following cammand:
```console
pip install FADS
```


# Diversity Subsampling 

[Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement), [Diversity Subsampling With Replacement](#diversity-subsampling-with-replacement)  show how to select a diverse subsample from a data set without and with replacement respectively. 

## Diversity Subsampling Without Replacement
In this section, we show how to use **FADS** to select a diverse subsample from a data set in Python 3. We will select a diverse subsample with size 2000 from the MGM data set under the default hyper-parameter setting. There are four additional hyper-parameters for the ’DS’ function: 
- n_components: the number of components to use for estimating the density of the data using a Gaussian Mixture Model (GMM). Default value is 32.
- init_params: method to initialize the component probabilities for GMM. It must be either ’kmeans’ or ’random’. When specified as ’kmeans’, the initial component probabilities in GMM will be chosen using the kmeans algorithm ([[3]](#3)); when specified as ’random’, the initial component probabilities in GMM will be chosen randomly. Default value is 'kmeans'.
- max_iter: maximum number of Expectation-Maximization (EM) iterations to perform to build the GMM. Default value is 10.
- update_iter: number of additional EM iterations to perform when updating the density regularly along the subsampling process. Note that in this case the previously resulting GMM parameters will be used as initial values. Default value is 1.


One can also use the built-in function in **FADS** to tune the hyper-parameters (see [Hyper-Parameter Tuning](#hyper-parameter-tuning) for details) or specify the values of n_components, init_params, max_iter, and update_iter explicitly. The python code to select a diverse subsample from a data set without replacement using the 'DS' function is shown below.

```python
#we will suppress warnings given by the sklearn GMM module due to convergence issues
import warnings
import ds_fns as df

#subsample size
nmax = 2000

#perform diversity subsampling
fastds = df.FADS(data)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ds_idx = fastds.DS(nmax)

#the coordinates the selected subsample
ds_sample = data[ds_idx,:]
```

The returned Numpy array ds_idx contains the selected indices of each subsample point; the selected DS subsample is fully-sequential. We can plot the subsamples at various sizes as follows. In the below plot, n denotes the subsample size. The red open circles indicate selected subsample points and the gray dots represent a size-2000 random subset of the data.


<img src="https://github.com/boyangshang/FADS/blob/main/Graphs4Readme/2D_gmm_DS_norep_subsample.jpg" alt="DS subsample" width="850"/>

## Diversity Subsampling With Replacement
The DS_WR function in **FADS** selects a diverse subsample from a data set with replacement. There are three hyper-parameters for the ’DS_WR’ function: n_components, init_params and max_iter, the definitions of which are the same as in [Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement).


The following code shows how to use the DS_WR function to select a diverse subsample from a data set with replacement.
```python
#we will suppress warnings given by the sklearn GMM module due to convergence issues
import warnings
import ds_fns as df

#subsample size
nmax = 2000

#perform diversity subsampling
fastds = df.FADS(data)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    ds_idx = fastds.DS_WR(nmax)
    
#coordinates of the selected subsample
sample = data[ds_idx,:]
```


# Custom Subsampling

The DS_g function in **FADS** selects a custom subsample without replacement from a data set having some desired property other than/along with the diversity property. Compared with the DS function (see [Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement)), the DS_g function has two additional hyper-parameters:

- target_pdf_list: the desired subsampling ratios of each data point in the data set. It should be a numpy array of size N, where N is the data set size. 
- reg_param: a number in interval [0, 100]. It controls how diverse the selected custom subsample is. By design, the larger reg_param is, the more diverse the custom subsample will be; and vice versa. The DS_g function uses reg_param in the following way. For convenience, let reg_param = &alpha;. Suppose the desired subsampling ratio of each point in the data set D = {x<sub>1</sub>, &hellip;, x<sub>N</sub>} is {u(x<sub>1</sub>), &hellip;, u(x<sub>N</sub>)}. In the DS_g function, the subsampling ratio of each point is set as g(x<sub>i</sub>) = u(x<sub>i</sub>) + u<sub>&alpha;</sub>, where u<sub>&alpha;</sub> is the lower &alpha;&percnt; quantile of set {u(x<sub>1</sub>), &hellip;, u(x<sub>N</sub>)}, for i = 1, &hellip; ,N.

Now we provide an example using the DS_g function in **FADS** to select a custom subsample without replacement from the 2D MGM data set. The target subsampling ratio at each point is computed as the probability density value of the multivariate normal distribution with mean (-2.5, 2.5)<sup>T</sup> and a covariance matrix equaling the identity matrix in the Euclidean space of dimension 2. The code is as follows. As before, we use the default hyper-parameter setting here. [Hyper-Parameter Tuning](#hyper-parameter-tuning) discuss how to tune these hyper-parameters for a specific data set; one can also set the values of n_components, init_params, max_iter and update_iter explicitly. The following figure shows the selected custom subsamples with varying reg_param values at subsample size n = 200. The red open circles indicate selected subsample points; the small gray dots represent a random subset with a size 2000 of the data.

```python
import warnings
import ds_fns as df
import scipy.stats as ss
desired_ratios = ss.norm(-2.5, 1).pdf(data[:,0])*ss.norm(2.5, 1).pdf(data[:,1])

#subsampling using the DS_g function under the default hyper-paramter settings
nmax = 2000
reg_param = 50

#custom subsampling
fastds = df.FADS(data)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ds_g_idx = fastds.DS_g(nmax, target_pdf_list = desired_ratios, reg_param = reg_param)
    
#coordinates of selected subsample
mysubsample = data[ds_g_idx,:]
```


<img src="https://github.com/boyangshang/FADS/blob/main/Graphs4Readme/DSg_2D_gmm_DS_norep_subsample.jpg" alt="CS subsample" height="650"/>

# Hyper-Parameter Tuning
Hyper-parameters for methods in the **FADS** package are related to the process of estimating the probability density function evaluated at every point in the data set using GMM. The GMM density estimation procedure in **FADS** uses the ’GaussianMixture’ model in Scikit-learn ([[2]](#2)). The function for hyper-parameter tuning in **FADS** is tune_param_cv and below list its inputs and outputs. 

*
## Inputs
- ncomponent_list: a python list of possible choices of 'ncomponents'; see [Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement) for the definition of 'ncomponents'. Default value is [2,10,50].
- max_iter_list: a python list of possible choices of 'max_iter'; see [Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement) for the definition of 'max_iter'. Default value is [10, 50,100].
- nfold: integer; how many folds to use for the Cross-Validation(CV) procedure. Default value is 3.
- init_list: a python list of possible choices of 'init_params'; see [Diversity Subsampling Without Replacement](#diversity-subsampling-without-replacement) for the definition of 'init_params'. Default value is ['kmeans', 'random'].
- fraction: a float ranging from 0 to 1; a random subset of size &LeftFloor;fraction*N&RightFloor; will be selected from the data for the CV procedure. Here N denotes the data set size and &LeftFloor;fraction*N&RightFloor; denotes the largest integer not larger than N/2. Default value is 1.0.


## Outputs 
There is not output for the tune_params_cv function.
*

The tune_params_cv function follows the following algorithm to tune the hyper-parameters. Suppose that there are t = 1, &hellip; T different hyper-parameter settings to choose from.
- Randomly choose a subset of the entire data set with size &LeftFloor;fraction*N&RightFloor;;
- Do a k-fold CV (k = 3 by default and can be specified by the user) and compute the testing log-likelihood L<sub>t</sub> for t = 1, &hellip; T, using the data subset selected in the above step;
- Find the highest testing log-likelihood, say L<sub>max</sub>;
- Compute the computational-cost score of each hyper-parameter setting. For convenience, let iter<sub>max</sub> = max(max_iter_list) and let ncomp<sub>max</sub>  = max(ncomponent_list). For each hyper-parameter setting t, its computational-cost score is defined as C<sub>t</sub> = ncomponent<sub>t</sub>/ncomp<sub>max</sub> + max_iter <sub>t</sub>/iter<sub>max</sub>, t = 1, &hellip; T. Here ncomponent<sub>t</sub> and max_iter <sub>t</sub> are the ncomponent and max_iter values used for hyper-parameter setting t;
- Sort {C<sub>1</sub>, &hellip;, C<sub>T</sub>} is non-desending order and store the corresponding indices as j<sub>1</sub>, &hellip;, j<sub>T</sub>, such that C<sub>j<sub>m</sub></sub> <= C<sub>j<sub>n</sub></sub>, as long as 1 &le; m &le; n &le; T;
- Find the smallest t such that |L<sub>max</sub>-L<sub>j<sub>t</sub></sub>|/L<sub>max</sub> < 1&percnt;,  t = 1, &hellip; T; denote it as t<sub>best</sub>;
- The hyper-parameter setting with index j<sub>t<sub>best</sub></sub> will be chosen as the best one.

By [[1]](#1), setting ncomponent = 32, max_iter = 10, update_iter = 1, and init_params = ’kmeans’ works well for all tested examples with various data distributions in 2D and 10D in their experiments, including product forms of standard normal, exponential, gamma, geometric distributions, and a mixture of multivariate Gaussian distributions. So we use this setting as the default hyper-parameter setting for the **FADS** package. 

For the choice of ’update_iter’, larger values of ’update_iter’ usually lead to better accuracy in density updating, at the price of longer runtime. Since in the DS algorithm, previously obtained GMM parameters are used as initial guesses for the updating process, we suggest using a smaller value for update_iter than max_iter for better computational efficiency. The default setting is update_iter = 1. Note that ’update_iter’ will not be tuned by the tune_params_cv function and the user is expected to use the default setting or to specify it explicitly.

The code for hyper-paramter tuning using tune_params_cv is shown below. Here the possible choices of ’ncomponent’, ’max_iter’ and ’init_params’ are respectively {2, 15, 30}, {10, 100} and {’kmeans’}. We use a random subset of the data with size &LeftFloor;N/2&RightFloor; to perform this task.

```python
import ds_fns as df

fastds = df.FADS(data, tune_params = True)
#user-specified parameter choices
nfold = 3
ncomponents = [2,15,30]
max_iters = [10,100]
inits = ['kmeans']
fraction = 0.5

#tune hyper-parameters
fastds.tune_params_CV(ncomponent_list = ncomponents, max_iter_list = max_iters, init_list = inits, nfold = nfold,fraction = fraction)

#e.g. use DS to select a diverse subsample with size 2000 using the tuned paramters
n = 2000
ds_idx = fastds.DS(n)
```

# References
<a id="1">[1]</a> 
Shang, B. and Apley, D. and Mehrotra, S (2022). 
Diversity Subsampling: Custom Subsamples from a Data Set. 
submitted to Informs Journal of Data Science, 00(0), pp-pp.

<a id="2">[2]</a> 
Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E. (2011). 
         Scikit-learn: Machine Learning in {P}ython.
            Journal of Machine Learning Research, 12, 2825-2830.
            
<a id="3">[3]</a> 
MacQueen, J. (1967). 
Some methods for classification and analysis of multivariate observations. 
Proceedings of the fifth Berkeley symposium on mathematical statistics and probability, 1(14), 281-297.




# Appendix A: Synthetic Data Used in this Article
The synthetic data used to obtain results in this article can be generated using the following code.

```python
import numpy as np

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
```
