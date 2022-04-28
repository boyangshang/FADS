This mannual provides examples using the FAst Diversity Subsampling (**FADS**) package developed by [[1]](#1) to select diverse subsamples (**DS**) and custom subsamples (**CS**) from a data set in Python. As discussed in [[1]](#1), a diverse subsample is a subset of a data set that is spread out over the region occupied by the (usually unknown) data distribution. A custom subsample is selected proportional to the user-defined target sampling ratios, with or without consideration of the diversity property.

The experimental setting for this article is as follows. For easy visualization, throughout this article, we use a synthetic Multivariate Gaussian Mixture (MGM) data in 2D. Details about generating the data are provided in [Appendix A](#appendix-A:-synthetic-data-used-in-this-article). For tuning the hyper-parameters, the default setting has been numerically verified to work well for a variety of data distributions up to dimension 10 (see [[1]](#1)). The **FADS** package offers a built-in function to further tune the hyper-parameters; please see [Hyper-Parameter Tuning](#hyper-parameter-tuning) for details. Default settings will be used for examples in this article unless otherwise specified.

This article is organized in the following way. [Diversity Subsampling](#diversity-subsampling) shows how to use **FADS** to select a diverse subsample from a data set without or with replacement. [Custom Subsampling](#custom-subsampling) provides one example to use **FADS** to select a custom subsample. [Hyper-Parameter Tuning](#hyper-parameter-tuning) shows how to use the built-in function in **FADS** to further tune the hyper-parameters for either the **DS** or **CS** method.


# Diversity Subsampling 

In this section, we show how to use **FADS** to select a diverse subsample from a data set in Python 3. We will select a diverse subsample from the MGM data set with size <img src="http://www.sciweavers.org/tex2img.php?eq=nmax%20%3D%202000&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="nmax = 2000" width="117" height="15" /> nmax = 2000 under the default parameter settings. There are four additional hyper-parameters for the ’DS’ function:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bpmatrix%7D%0Ao%20%26%201%5C%5C%0A1%20%26%200%0A%5Cend%7Bpmatrix%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\begin{pmatrix}o & 1\\1 & 0\end{pmatrix}" width="58" height="39" />

<img src="https://latex.codecogs.com/svg.image?\bg{white}\begin{pmatrix}o&space;&&space;1\\1&space;&&space;0\end{pmatrix}" title="https://latex.codecogs.com/svg.image?\bg{white}\begin{pmatrix}o & 1\\1 & 0\end{pmatrix}" />

# Custom Subsampling

# Hyper-Parameter Tuning

# References
<a id="1">[1]</a> 
Shang, B. and Apley, D. and Mehrotra, S (2022). 
Diversity Subsampling: Custom Subsamples from a Data Set. 
submitted to Informs Journal of Data Science, 00(0), pp-pp.

# Appendix A: Synthetic Data Used in this Article

