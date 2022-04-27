In this article we provide examples using the FAst Diversity Subsampling (**FADS**) package developed by [[1]](#1) to select diverse subsamples (**DS**) and custom subsamples (**CS**) from a data set in Python. As discussed in [[1]](#1), a diverse subsample is a subset of a data that is spread-out over the region occupied by the (usually unknown) data distribution. A custom subsample is selected proportional to the user-defined target sampling ratios, with or without consideration of the diversity property.

The experimental setting for this article is as follows. For the purpose of easy visualization, throughout this article, we use a synthetic Multivariate Gaussian Mixture (MGM) data in 2D. Details about generating the data are provided in [Appendix A](#appendix-A:-synthetic-data-used-in-this-article). For tuning the hyper-parameters, the default setting has been numerically verified to work well for a variety of data distributions
up to dimension 10 (see [[1]](#1)). The **FADS** packages offers a built-in function to further tune the hyper-parameters; please see [Hyper-Parameter Tuning](#hyper-parameter-tuning) for details. Default settings will be used for examples in this article unless otherwise specified.

This article is organized in the following way. [Diversity Subsampling](#diversity-subsampling) shows how to use **FADS** to select a diverse subsample from a data set without or with replacement. [Custom Subsampling](#custom-subsampling) provides one example to use **FADS** to select a custom subsample. [Hyper-Parameter Tuning](#hyper-parameter-tuning) shows how to use the built-in function in **FADS** to further tune the hyper-parameters for either the **DS** or **CS** method.

# Diversity Subsampling 

# Custom Subsampling

# Hyper-Parameter Tuning

## References

<a id="1">[1]</a> 
Shang, B. and Apley, D. and Mehrotra, S (2022). 
Diversity Subsampling: Custom Subsamples from a Data Set. 
Communications of the ACM, 11(3), 147-148.

# Appendix A: Synthetic Data Used in this Article

