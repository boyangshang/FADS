import math
import numpy as np 
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import mixture
from sklearn.model_selection import GridSearchCV
import warnings



class FADS:
    '''
    This class implements the Diversity Subsampling and Custom Subsampling methods
    proposed in paper <>
    
    Authors:
    Boyang Shang <boyangshang2015@u.northwestern.edu>
    Daniel Apley <apley@northwestern.edu>
    Sanjay Mehrotra <mehrotra@northwestern.edu>
    
    Last Updated on: Mar 09, 2022    
    
    '''
    
    def __init__(self,data,tune_params = False):
        '''
        Inputs:
        data - numpy array of size (N, dimension). Here N is the number of data points; dimension denotes the dimension fo the input space
        tune_params - bool. If True, the parameters of GMM density estiation will be chosen via CV instead of by defaults. One can specify possible choices of the number of components and the possible maximum number of iterations to use for GMM using the set_param_choices function. 
        '''
        
        self.data = data
        self.dimension = data.shape[1]
        self.N = data.shape[0]
        self.tune_params = tune_params
        self.best_params = {}
            
        
        
    def tune_params_CV(self, ncomponent_list = [2,10,50], max_iter_list = [10, 50,100], nfold = 3,init_list = ['kmeans', 'random'], fraction = 1.0):
        '''
        Tune parameters for GMM density estimation using CV
        
        Input:
        ncompnent_list - python list of integers, the number of components to use for GMM density estimation
        max_iter_list - pyton list of integers, the possible maximum number of iterations for EM updates during the GMM density estimation
        nfold - int, the number of folds to use for a nfold-CV
        init_list - python list of possible ways to initialize GMM component probabilities. Each element in init_list must be either 'kmeans' or 'random'.
        fraction - the portaion of the data to use for CV to tune parameters. For example, in the case when fraction = 0.5, a random subset of data with size self.N/2 will be used instead of self.data. 
        
        Output:
        No output.
        '''
        data = self.data
        N = self.N
        
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("The values of fraction should be in (0,1]")
        
        if fraction < 1.0:
            rand_idx = np.random.choice(range(N),int(N/2)+1,replace= True)
            data = data[rand_idx,:]
            
        if self.tune_params:
            #possible parameter choices for each paramter
            param_choices = {}
            #set paramter choices for CV for tunning
            #number of components GMM
            param_choices['n_components'] = sorted(ncomponent_list)
            #number of maximum number of iterms for GMM
            param_choices['max_iter'] = sorted(max_iter_list)
            #methods to initialize component ratios for GMM, there are only two choices so the users do not have to specify this one
            if not set(init_list).issubset(['kmeans','random']):
                raise ValueError("Each element in init_list must be either 'kmeans' or 'random'.")     
            param_choices['init_params'] = sorted(init_list)
            
            
            #CV
            gmm =  mixture.GaussianMixture()
            clf = GridSearchCV(gmm, param_choices,cv=nfold,refit = False)
            #we will supress warnings due to GMM failing to converge
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(data)
                
            #choose the best paramters using CV results
            nchoices = len(clf.cv_results_['mean_test_score'])
            idx_choice = np.array(range(nchoices))[(clf.best_score_-clf.cv_results_['mean_test_score'])/np.abs(clf.best_score_) < 10**(-2)][0]
            self.best_params = clf.cv_results_['params'][idx_choice]
            print(self.best_params)
        else:
            print("I will use default settings since tune_params was set to be False.")
            
    
        
        

    def DS(self, n, ncomponent = 32, max_iter = 10, update_iter = 1, init_params='kmeans'):
        """
        Input: 
        n - number of sample points to be selected
        ncomponent - the number of component to use for GMM
        max_iter - maximum number of iterations for GMM density estimation
        update_iter - number of iterations to do for GMM density updating
        init_params - Ways to initialize GMM parameters, must be 'kmeans' or 'random'


        Output:
        sample_idx - the array of all indicies that correspond to sample points selected from data



        Notes:
        the density is updated along the way. Density is estimated using the GMM method with diagonal
        covariance matrices. 

        """
        N = self.N
        data = self.data
        dimension = self.dimension
        #if the parameters were tuned by CV, use the results; otherwise using default or use-specified values
        if self.tune_params:
            ncomponent = self.best_params['n_components']
            max_iter = self.best_params['max_iter']
            init_params = self.best_params['init_params']
        
        
        
        #n_update = np.min([100,math.floor(n/10)])
        n_update = np.max([100,math.floor(n/10)])
        freq_update = math.floor(n/n_update)


        
        ##standardize the original data set to a unit hypercube
        Min_arr = np.percentile(data,0.5,axis =0)
        Max_arr = np.percentile(data,99.5,axis =0)
        perturbed_data = (data - Min_arr)/(Max_arr - Min_arr)

        ##perturb the standardized data set
        #figure out the variance of the normal random variable used for perturabtion. Perturbation variance is shared accross dimensions.
        sigma_perturbation = 0.0

        while sigma_perturbation == 0.0:
            nrand = np.min([2000, int(N/4)])
            (unique, counts) = np.unique(perturbed_data[np.random.choice(range(N),nrand,replace = False),:], return_counts=True,axis = 0)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(unique)
            distances, indices = nbrs.kneighbors(unique)
            sigma_perturbation = np.min(distances[:,1])/8.0

        perturbed_data += np.random.normal(0,sigma_perturbation, data.shape)

        if freq_update > 0:
            rem_idx = np.array(range(0,N,1))
            rem_set = set(rem_idx)
            sample_idx=[]
            for i in range(1,int(freq_update+1),1):
                if i==1:

                    est_density_array, gmm = self.GMMdiag_density(perturbed_data, ncomponent, max_iter,init_params)
                    density = 1/est_density_array
                    sums = np.sum(density)
                    density = density/sums
                    selected_idx = np.random.choice(rem_idx, size=n_update, replace=False, p=density)
                    sample_idx.extend(selected_idx)
                    #rem_idx = list(set(rem_idx) - set(selected_idx))
                    rem_set -= set(selected_idx)
                    rem_idx = sorted(rem_set)
                    #density_list_full = np.copy(est_density_array)
                else:

                    est_density_array, gmm = self.GMM_diag_density_update(gmm, perturbed_data, rem_idx, update_iter)
                    density = 1/est_density_array
                    sums = np.sum(density)
                    density = density/sums
                    selected_idx = np.random.choice(rem_idx, size=n_update, replace=False, p=density)
                    sample_idx.extend(selected_idx)
                    rem_set -= set(selected_idx)
                    rem_idx = sorted(rem_set)

            if len(sample_idx) < n:

                nadd = n - len(sample_idx)


                est_density_array, gmm = self.GMM_diag_density_update(gmm, perturbed_data, rem_idx, update_iter)
                density = 1/est_density_array
                sums = np.sum(density)
                density = density/sums
                selected_idx = np.random.choice(rem_idx, size=nadd, replace=False, p=density)
                sample_idx.extend(selected_idx)

        else:
            est_density_array, gmm = self.GMMdiag_density(perturbed_data, ncomponent, max_iter,init_params)
            density = 1/est_density_array
            sums = np.sum(density)
            density = density/sums
            sample_idx = np.random.choice(range(N), size=n, replace=False, p=density)


        return sample_idx 

    def GMMdiag_density(self, data, ncomponent, max_iter,init_params):
        '''
        Estimate density of data using GMM 
        
        Input: 
        data - the data set to estimate the density for; this could be different from self.data due to perturbation
        ncomponent - the number of component to use for GMM
        max_iter - maximum number of iterations for GMM density estimation
        init_params - Ways to initialize GMM parameters, must be 'kmeans' or 'random'
        
        Output:
        gmm_density - numpy array of np.float64 with size N. The estimated density of self.data
        gmm - the constructed GaussianMixture class by sklearn
        '''
        
        N = data.shape[0]
        dimension = data.shape[1]
       
        

        gmm = mixture.GaussianMixture(n_components=ncomponent,covariance_type='diag',max_iter=max_iter,init_params = init_params,warm_start = False)
        gmm.fit(data)
        gmm_density = np.exp(gmm.score_samples(data))

        return gmm_density, gmm

    def GMM_diag_density_update(self, gmm, data, remaining_idx, update_iter):
        '''
        Re-Estimate density of remaining data.
        
        Input: 
        gmm - the constructed GaussianMixture class by sklearn
        data - the original data set to estimate the density for in GMMdiag_density; this could be different from self.data due to perturbation
        remaining_idx - the indices of the remaining data points after subsampling
        update_iter - number of iterations to do for GMM density updating
        
        
        Output:
        gmm_density - numpy array of np.float64 with size N. The estimated density of self.data
        gmm - the updated GaussianMixture class by sklearn
        '''      
        
        gmm.warm_start=True
        gmm.max_iter=update_iter
        #copying remaining data
        rem_data = data[remaining_idx,:]
        gmm.fit(rem_data)
        gmm_density = np.exp(gmm.score_samples(rem_data))

        return gmm_density, gmm



    def DS_g(self, n, target_pdf_list = None, ncomponent = 32, max_iter = 10, update_iter = 1, reg_param = 0.0, init_params='kmeans'):


        """
        Subsampling from the data set according to a target density function. Note that the target needs to be nonnegative, but it does not have to be a well-defined density function.

        Inputs: 


        n - number of subsample points to be selected


        target_pdf_list - numpy array of size N consisting the desired sampling ratio of each data point.  

        ncomponent - number of components used in GMM

        max_iter - maxmimum number of iterations for initial GMM density estimation

        update_iter - additional iteratinos run at each density updating step

        reg_param - nonnegative float between 0 and 100. the target density would be g/f + alpha/f, where g is specified via target_pdf_value. alpha is the lower reg_param*0.01 quantile of g evaluated on the data. The larger reg_param is, the more space-filling the subsample would be.

        init_params - Ways to initialize GMM parameters, must be 'kmeans' or 'random'



        Outputs:
        sample_idx - the array of all indicies that correspond to sample points selected from data


        Notes:
        the density is updated along the way. 

        We add a normal perturbation to the data sets to void replicated data points and coordinates for robustness.
        """
        N = self.N
        data = self.data
        dimension = self.dimension
        #if the parameters were tuned by CV, use the results; otherwise using default or use-specified values
        if self.tune_params:
            ncomponent = self.best_params['n_components']
            max_iter = self.best_params['max_iter']
            init_params = self.best_params['init_params']
            
            
        #figure out how to get the target subsampling ratios 
        
        if (target_pdf_list is None):
            print('No target sampling ratios were provided. I will select a diverse subsample then.')
            target_pdf_list = np.ones(N, dtype = np.float64)

   
        n_update = np.max([100,math.floor(n/10)])
        freq_update = math.floor(n/n_update)


        ##standardize the original data set to a unit hypercube

        Min_arr = np.percentile(data,0.5,axis =0)
        Max_arr = np.percentile(data,99.5,axis =0)
        perturbed_data = (data - Min_arr)/(Max_arr - Min_arr)

        ##perturb the standardized data set
        #figure out the variance of the normal random variable used for perturabtion. Perturbation variance is shared accross dimensions.
        sigma_perturbation = 0.0

        while sigma_perturbation == 0.0:
            nrand = np.min([2000, int(N/4)])
            (unique, counts) = np.unique(perturbed_data[np.random.choice(range(N),nrand,replace = False),:], return_counts=True,axis = 0)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(unique)
            distances, indices = nbrs.kneighbors(unique)
            sigma_perturbation = np.min(distances[:,1])/8.0

        perturbed_data += np.random.normal(0,sigma_perturbation, data.shape)
        
        #subsampling
        sample_idx=[]

        if freq_update > 0:
            rem_idx = np.array(range(0,N,1))
            rem_set = set(rem_idx)

            for i in range(1,int(freq_update+1),1):
                if i==1:

                    est_density_array, gmm = self.GMMdiag_density(perturbed_data, ncomponent, max_iter,init_params)
                    #target pdf vlues                    
                    g_list = target_pdf_list   
                    #we do not allow too small density values which might be due to numerical errors
                    tinynumber = np.finfo(np.float64).eps 
                    idx = g_list < tinynumber
                    g_list[idx] = tinynumber
                    myalpha =np.percentile(g_list, reg_param)                
                    density = (g_list+myalpha)/est_density_array             
                    sums = np.sum(density)
                    density = density/sums
                    selected_idx = np.random.choice(rem_idx, size=n_update, replace=False, p=density)
                    sample_idx.extend(selected_idx)
                    #rem_idx = list(set(rem_idx) - set(selected_idx))
                    rem_set -= set(selected_idx)
                    rem_idx = sorted(rem_set)
                    #density_list_full = np.copy(est_density_array)
                else:

                    est_density_array, gmm = self.GMM_diag_density_update(gmm, perturbed_data, rem_idx, update_iter)
                    
                    g_list = target_pdf_list[rem_idx]                   
                    #we do not allow too small density values which might be due to numerical errors
                    tinynumber = np.finfo(np.float64).eps 
                    idx = g_list < tinynumber
                    g_list[idx] = tinynumber
                    myalpha =np.percentile(g_list, reg_param) 
                    density = (g_list+myalpha)/est_density_array
                    sums = np.sum(density)
                    density = density/sums
                    selected_idx = np.random.choice(rem_idx, size=n_update, replace=False, p=density)
                    sample_idx.extend(selected_idx)
                    #rem_idx = list(set(rem_idx) - set(selected_idx))
                    rem_set -= set(selected_idx)
                    rem_idx = sorted(rem_set)

            if len(sample_idx) < n:

                nadd = n - len(sample_idx)
                est_density_array, gmm = self.GMM_diag_density_update(gmm, perturbed_data, rem_idx, update_iter)               
                g_list = target_pdf_list[rem_idx]                
                #we do not allow too small density values which might be due to numerical errors
                tinynumber = np.finfo(np.float64).eps 
                idx = g_list < tinynumber
                g_list[idx] = tinynumber
                myalpha =np.percentile(g_list, reg_param) 
                density = (g_list+myalpha)/est_density_array
                sums = np.sum(density)
                density = density/sums
                selected_idx = np.random.choice(rem_idx, size=nadd, replace=False, p=density)
                sample_idx.extend(selected_idx)

        else:
            est_density_array, gmm = self.GMMdiag_density(perturbed_data, ncomponent, max_iter,init_params)
            #target pdf vlues            
            g_list = target_pdf_list            
            #we do not allow too small density values which might be due to numerical errors
            tinynumber = np.finfo(np.float64).eps 
            idx = g_list < tinynumber
            g_list[idx] = tinynumber
            myalpha =np.percentile(g_list, reg_param)                
            density = (g_list+myalpha)/est_density_array
            sums = np.sum(density)
            density = density/sums
            sample_idx = np.random.choice(range(N), size=n, replace=False, p=density)


        return sample_idx 

    def DS_WR(self, n, ncomponent = 32, max_iter = 10, init_params='kmeans'):
        """
        Input: 
        n - number of sample points to be selected
        ncomponent - the number of component to use for GMM
        max_iter - maximum number of iterations for GMM density estimation
        init_params - Ways to initialize GMM parameters, must be 'kmeans' or 'random'


        Output:
        sample_idx - the array of all indicies that correspond to sample points selected from data

        """
        N = self.N
        data = self.data
        dimension = self.dimension
        #if the parameters were tuned by CV, use the results; otherwise using default or use-specified values
        if self.tune_params:
            ncomponent = self.best_params['n_components']
            max_iter = self.best_params['max_iter']
            init_params = self.best_params['init_params']




        ##standardize the original data set to a unit hypercube
        Min_arr = np.percentile(data,0.5,axis =0)
        Max_arr = np.percentile(data,99.5,axis =0)
        perturbed_data = (data - Min_arr)/(Max_arr - Min_arr)

        ##perturb the standardized data set
        #figure out the variance of the normal random variable used for perturabtion. Perturbation variance is shared accross dimensions.
        sigma_perturbation = 0.0

        while sigma_perturbation == 0.0:
            nrand = np.min([2000, int(N/4)])
            (unique, counts) = np.unique(perturbed_data[np.random.choice(range(N),nrand,replace = False),:], return_counts=True,axis = 0)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(unique)
            distances, indices = nbrs.kneighbors(unique)
            sigma_perturbation = np.min(distances[:,1])/8.0

        perturbed_data += np.random.normal(0,sigma_perturbation, data.shape)

        est_density_array, gmm = self.GMMdiag_density(perturbed_data, ncomponent, max_iter,init_params)
        #print(est_density_array[0:7])
        density = 1/est_density_array
        sums = np.sum(density)
        density = density/sums
        sample_idx = np.random.choice(range(N), size=n, replace=True, p=density)



        return sample_idx 

