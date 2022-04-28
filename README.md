
This mannual provides examples using the FAst Diversity Subsampling (**FADS**) package developed by [[1]](#1) to select diverse subsamples (**DS**) and custom subsamples (**CS**) from a data set in Python. As discussed in [[1]](#1), a diverse subsample is a subset of a data set that is spread out over the region occupied by the (usually unknown) data distribution. A custom subsample is selected proportional to the user-defined target sampling ratios, with or without consideration of the diversity property.

The experimental setting for this article is as follows. For easy visualization, throughout this article, we use a synthetic Multivariate Gaussian Mixture (MGM) data in 2D. Details about generating the data are provided in [Appendix A](#appendix-A:-synthetic-data-used-in-this-article). For tuning the hyper-parameters, the default setting has been numerically verified to work well for a variety of data distributions up to dimension 10 (see [[1]](#1)). The **FADS** package offers a built-in function to further tune the hyper-parameters; please see [Hyper-Parameter Tuning](#hyper-parameter-tuning) for details. Default settings will be used for examples in this article unless otherwise specified.

This article is organized in the following way. [Diversity Subsampling](#diversity-subsampling) shows how to use **FADS** to select a diverse subsample from a data set without or with replacement. [Custom Subsampling](#custom-subsampling) provides one example to use **FADS** to select a custom subsample. [Hyper-Parameter Tuning](#hyper-parameter-tuning) shows how to use the built-in function in **FADS** to further tune the hyper-parameters for either the **DS** or **CS** method.

# Installation 
The **FADS** package is available on pip and can be installed using the following cammand:
```console
pip install FADS
```


# Diversity Subsampling 

In this section, we show how to use **FADS** to select a diverse subsample from a data set in Python 3. We will select a diverse subsample with size 2000 from the MGM data set under the default parameter settings. There are four additional hyper-parameters for the ’DS’ function: 
- n_components: the number of components to use for estimating the density of the data using a Gaussian Mixture Model (GMM)
- init_params: method to initialize the component probabilities for GMM. It must be either ’kmeans’ or ’random’
- max_iter: maximum number of Expectation-Maximization (EM) iterations to perform to build the GMM
- update_iter: number of additional EM iterations to perform when updating the density regularly along the subsample
selection process. Note that in this case the previously resulting GMM parameters will be used as initial values.

One can also use the built-in function in **FADS** to tune the hyper-parameters (see [Hyper-Parameter Tuning](#hyper-parameter-tuning) for details) using maximum likelihood estimators or specify the values of n_components, init_params, max_iter, and update_iter explicitly. Subsampling without replacement is generally recommended and the code is shown below.

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

The returned Numpy array ds_idx contains the selected indices of each subsample point; the selected DS subsample is fully-sequential. We can plot the subsamples at various sizes as follows. In the below plot, n denotes the subsample size and the subsample is fully-sequential. 

n = 20            |  n = 150 |n = 450
:-------------------------:|:-------------------------:|:-------------------------
![alt text](https://github.com/boyangshang/FADS/blob/main/Graphs4Readme/sea-subsample/2D_gmm_n_50_DS_norep_subsample.pdf)  |  ![alt text](https://github.com/boyangshang/FADS/blob/main/Graphs4Readme/sea-subsample/2D_gmm_n_150_DS_norep_subsample.pdf) | ![alt text](https://github.com/boyangshang/FADS/blob/main/Graphs4Readme/sea-subsample/2D_gmm_n_450_DS_norep_subsample.pdf)

# Custom Subsampling

# Hyper-Parameter Tuning

# References
<a id="1">[1]</a> 
Shang, B. and Apley, D. and Mehrotra, S (2022). 
Diversity Subsampling: Custom Subsamples from a Data Set. 
submitted to Informs Journal of Data Science, 00(0), pp-pp.

# Appendix A: Synthetic Data Used in this Article

