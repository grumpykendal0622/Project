""" Plot and log results of the experiments on simulated data """

import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_factors(W, model, total_var):
    """
    Find the shared and specific latent factors.

    Parameters
    ----------
    W : array-like, shape (n_features, n_comps)
        Concatenated loading matrix. The number of features
        corresponds to the total number of features in all groups.

    model : object
        The model output containing the 's' and 'd' attributes.

    total_var : float
        Total variance explained.

    Returns
    -------
    relfactors_shared : dict
        A dictionary where keys are tuples of group indices and values are lists
        of shared factors indices.

    relfactors_specific : list
        A list of the relevant factors specific to each group.
    """
    ncomps = W.shape[1]
    var_within = np.zeros((model.s, ncomps))
    d = 0
    for s in range(model.s):
        Dm = model.d[s]
        for c in range(ncomps):
            var_within[s, c] = np.sum(W[d:d + Dm, c] ** 2) / total_var * 100
        d += Dm

    relvar_within = np.zeros((model.s, ncomps))
    for s in range(model.s):
        relvar_within[s, :] = var_within[s, :] / np.sum(var_within[s, :]) * 100

    relfactors_specific = [[] for _ in range(model.s)]
    relfactors_shared = {}
    
            # if np.any(relvar_within[:,c] > 7.5):
            #     if ratio > 300:
            #     relfactors_specific[1].append(c)
            # elif ratio < 0.001:
            #     relfactors_specific[0].append(c)
            # else:
            #     relfactors_shared.append(c) 

    for c in range(ncomps):
        shared_indices = []
        for s in range(model.s):
            if relvar_within[s, c] > 7.5:
                shared_indices.append(s)
        if shared_indices:  # Check if shared_indices is not empty
            if len(shared_indices) > 1:
                if len(shared_indices) == model.s:
                    # This factor is shared across all groups
                    print(f"This factor {c + 1} is shared across all groups")
                    print(shared_indices)
        
                    relfactors_shared[tuple(shared_indices)] = relfactors_shared.get(tuple(shared_indices), []) + [c]
                else:
                    print(f"This factor {c + 1} is shared across some but not all groups")
                    print(shared_indices)
                    # This factor is shared across some but not all groups
                    relfactors_shared[tuple(shared_indices)] = relfactors_shared.get(tuple(shared_indices), []) + [c]
            else:
                # This factor is specific to one group
                relfactors_specific[shared_indices[0]].append(c)
                
        # Filter out empty lists from relfactors_specific
    relfactors_specific = [factors for factors in relfactors_specific if factors]


    return relfactors_shared, relfactors_specific

def get_results(args, res_dir, missing_info=None):  

    """ 
    Plot and save the results of the experiments on synthetic data.

    Parameters
    ----------   
    args : local namespace 
        Arguments selected to run the model.

    res_dir : str
        Path to the directory where the results will be saved.

    missing_info : dict | None, optional.
        Parameters to generate data with missing values.         

    """  

    # Number of runs
    num_runs = args.num_runs

    # Initialize variables to store metrics
    MSE = np.zeros(num_runs)
    MSE_chance_level = np.zeros(num_runs)
    MSE_median = np.zeros(num_runs) if args.impMedian else None
    ELBO = np.zeros(num_runs)
    Corr_miss = np.zeros((len(missing_info['ds']), num_runs)) if 'incomplete' in args.scenario else None
    
    ofile = open(f'{res_dir}/results.txt','w')   
    
    for i in range(0, num_runs):       
        print(f'Run: {i+1}', file=ofile)

        # Load model output
        with open(f'{res_dir}/[{i+1}]ModelOutput.dictionary', 'rb') as parameters:
            GFAotp = pickle.load(parameters)

        # Load median model output if applicable
        if args.impMedian:
            with open(f'{res_dir}/[{i+1}]ModelOutput_median.dictionary', 'rb') as parameters:
                GFAotp_median = pickle.load(parameters)
   
        # Print and store ELBO
        ELBO[i] = GFAotp.L[-1]
        print(f'ELBO (last value): {ELBO[i]:.2f}', file=ofile)

        # Print inferred taus
        for m in range(args.num_groups):
            inferred_tau = GFAotp.E_tau[0, m] if 'spherical' in args.noise else np.mean(GFAotp.E_tau[m])
            print(f'Inferred taus (group {m+1}): {inferred_tau:.2f}', file=ofile)
            
        # Get and store predictions
        MSE[i] = GFAotp.MSE
        MSE_chance_level[i] = GFAotp.MSE_chlev
        if args.impMedian:
            MSE_median[i] = GFAotp_median.MSE

        # Store correlation for missing values if applicable
        if 'incomplete' in args.scenario:
            for j, ds in enumerate(missing_info['ds']):
                Corr_miss[j, i] = GFAotp.Corr_miss[0, j]              
    

    # Plot results for the best run
    # -------------------------------------------------------------------------------
    # Identify the best run based on the highest ELBO
    best_run = np.argmax(ELBO)
    
    # Record the overall results
    print('\nOVERALL RESULTS--------------------------', file=ofile)   
    print(f'BEST RUN: {best_run + 1}', file=ofile)
    print(f'MSE: {MSE[best_run]:.2f}', file=ofile)


    # Load the model output and data files of the best run
    with open(f'{res_dir}/[{best_run + 1}]ModelOutput.dictionary', 'rb') as file:
        GFAotp_best = pickle.load(file)
    with open(f'{res_dir}/[{best_run + 1}]Data.dictionary', 'rb') as file:
        data = pickle.load(file)


    # Plot true and inferred parameters
    plot_params(GFAotp_best, res_dir, args, best_run, data, plot_trueparams=True) 
    
    # Initialize matrices to store true and inferred parameters
    T = np.zeros(np.sum(GFAotp_best.d))
    W = np.zeros((np.sum(GFAotp_best.d), GFAotp_best.k))
    W_true = np.zeros((np.sum(GFAotp_best.d), data['true_K']))

    # Construct the T and W matrices
    d = 0
    for m in range(args.num_groups):
        Dm = GFAotp_best.d[m]
        tau_m = GFAotp_best.E_tau[m] if 'diagonal' in args.noise else GFAotp_best.E_tau[0, m]
        T[d:d + Dm] = 1 / tau_m
        W[d:d + Dm, :] = GFAotp_best.means_w[m]
        W_true[d:d + Dm, :] = data['W'][m]
        d += Dm

    # Convert T into a diagonal matrix
    T = np.diag(T)

    # Check if the number of true and inferred factors match
    if GFAotp_best.k == data['true_K']:
        # Match true and inferred factors and calculate correlations
        W, _, _, factor_correlations = match_factors(W, W_true)
        print(f'Similarity of the factors (Pearsons correlation): {factor_correlations}', file=ofile)
    Est_totalvar = np.trace(np.dot(W, W.T) + T)
    true_totalvar = np.trace(np.dot(W_true, W_true.T))
    print(f'\nTotal variance explained by the true factors: {true_totalvar:.2f}', file=ofile)
    print(f'Total variance explained by the inferred factors: {Est_totalvar:.2f}', file=ofile)


    if GFAotp_best.k == data['true_K']:
        shared, specific = get_factors(W, GFAotp_best, Est_totalvar)
        print("specific")
        print(specific)
    
        # Print shared factors
        for groups, factors in shared.items():
            groups_str = ', '.join([str(g + 1) for g in groups])  # Add 1 to group index for display
            print(f'Factor {np.array(factors) + 1} is shared among groups {groups_str}.', file=ofile)

        # Print specific factors for each group
        for m, factors in enumerate(specific):
            print(f'Factor {np.array(factors) + 1} is specific for group group {m + 1}. ' ,file=ofile)
    else:
        print("Incorrect Z has been inferred.", file=ofile)


    # Begin a section on multi-output predictions
    print('\nMulti-output predictions-----------------', file=ofile)
    
    # Display the Mean Squared Error (MSE) of the model when only observed data is used
    # along with the standard deviation of the MSE across multiple runs
    mean_mse = np.mean(MSE)
    std_mse = np.std(MSE)
    print('Model with observed data only:', file=ofile)
    print(f'MSE (avg(std)): {mean_mse:.2f} ({std_mse:.3f})', file=ofile)
    
    # Display the MSE at chance level, which serves as a baseline comparison
    # The average and standard deviation of the MSE at chance level.
    mean_MSE_chance_level = np.mean(MSE_chance_level)
    std_MSE_chance_level = np.std(MSE_chance_level)
    print('Chance level:', file=ofile)
    print(f'MSE (avg(std)): {mean_MSE_chance_level:.2f} ({std_MSE_chance_level:.3f})', file=ofile)
    
    # If the scenario includes incomplete data, provide additional metrics
    if 'incomplete' in args.scenario:
        print('\nPredictions for missing data -----------------', file=ofile)
        
        # Loop over each dataset with missing information
        for j, dataset_index in enumerate(missing_info['ds']):
            # Adjust index for zero-based indexing
            g_miss = dataset_index - 1
            
            # Print out which group (dataset) is being referred to
            print(f'Group: {g_miss + 1}', file=ofile)
            
            # Load the best run data dictionary to get the training data
            data_file = f'{res_dir}/[{best_run+1}]Data.dictionary'
            with open(data_file, 'rb') as parameters:
                data_best = pickle.load(parameters)
            
            # Calculate the percentage of missing data for this group
            X = data_best['X_tr'][g_miss]
            perc_miss = (np.isnan(X).sum() / X.size) * 100
            print(f'Percentage of missing data (group {g_miss + 1}): {perc_miss:.2f}%', file=ofile)
            
            # Display the average and standard deviation of the correlation for the missing data predictions
            mean_corr_miss = np.mean(Corr_miss[j, :])
            std_corr_miss = np.std(Corr_miss[j, :])
            print(f'Correlation (avg(std)): {mean_corr_miss:.3f} ({std_corr_miss:.3f})', file=ofile)


    # Begin a new section in the results file dedicated to models using median imputation
    if args.impMedian:
        print('\nModel with median imputation------------', file=ofile)
            
            # Load the model output for the best run that used median imputation
        model_median_file = f'{res_dir}/[{best_run+1}]ModelOutput_median.dictionary'
        with open(model_median_file, 'rb') as parameters:
            GFAotp_median_best = pickle.load(parameters)
            
            # Iterate through each group to print the inferred tau values
            # Tau values are model parameters related to the precision of the data
        for m in range(args.num_groups):
            tau_m = GFAotp_median_best.E_tau  # Extract the tau values from the model
                
                # Check if tau is structured to have more than one value per group
            if tau_m[0].size > args.num_groups:
                # If so, calculate and print the average tau for each group
                avg_tau = np.mean(tau_m[0][:, m])
                print(f'Inferred avg. taus (group {m+1}): {avg_tau:.2f}', file=ofile)
            else:
                # If not, print the single tau value for each group
                print(f'Inferred tau (group {m+1}): {tau_m[0, m]:.2f}', file=ofile)
            
        # Plot the parameters inferred from the best run with median imputation
        # The function 'plot_params' is assumed to create and save the plots
        plot_params(GFAotp_median_best, res_dir, args, best_run, data, plot_medianparams=True)
            
        # Print the performance of the model using median imputation
        # Specifically, log the average and standard deviation of the Mean Squared Error (MSE)
        avg_mse_med = np.mean(MSE_median)
        std_mse_med = np.std(MSE_median)
        print('Predictions:', file=ofile)
        print(f'MSE (avg(std)): {avg_mse_med:.3f} ({std_mse_med:.3f})', file=ofile)

    print('Visualisation concluded!')                               
    ofile.close() 
 


############ Function 'hinton_diag', 'match_factors', 'plot_Z' and 'plot_params' are from https://github.com/ferreirafabio80/gfa/blob/master/visualization_syntdata.py ###########
############ For the code simplicity it is directly displayed below.############
def hinton_diag(matrix, path):

    """ 
    Draw Hinton diagram for visualizing a weight matrix.

    Parameters
    ----------
    matrix : array-like 
        Weight matrix.
    
    path : str
        Path to save the diagram. 
    
    """
    plt.figure()
    ax = plt.gca()
    fcolor = 'white'
    max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor(fcolor)
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    plt.savefig(path)
    plt.close()

def match_factors(tempW, W_true):
    
    """ 
    Match the inferred factors to the true generated ones.

    Parameters
    ----------
    tempW : array-like, shape(n_features, n_comps)
        Concatenated inferred loading matrix. The number 
        of features here correspond to the total number of 
        features in all groups.

    W_true : array-like, shape(n_features, n_comps)
        Concatenated true loading matrix. The number of 
        features here correspond to the total number of 
        features in all groups.     

    Returns
    -------
    W : array-like, shape(n_features, n_comps)
        Sorted version of the inferred loading matrix.

    sim_factors : array-like, shape(n_comps,)
        Matching indices. These are obtained by calculating
        the Pearsons correlation between inferred and
        true factors.

    flip : list
        Flip sign info. Positive correlation corresponds
        to the same sign and negative correlation 
        represents inverse sign.

    maxcorr : list
        Maximum correlation between inferred and true 
        factors.

    """
    # Calculate similarity between the inferred and
    # true factors (using pearsons correlation)
    corr = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            corr[j,k] = np.corrcoef([W_true[:,k]],[tempW[:,j]])[0,1]
    sim_factors = np.argmax(np.absolute(corr),axis=0)
    maxcorr = np.max(np.absolute(corr),axis=0)
    
    # Sort the factors based on the similarity between inferred and
    #true factors.
    sim_thr = 0.70 #similarity threshold 
    sim_factors = sim_factors[maxcorr > sim_thr] 
    flip = []
    W = np.zeros((tempW.shape[0],sim_factors.size))
    for comp in range(sim_factors.size):
        if corr[sim_factors[comp],comp] > 0:
            W[:,comp] = tempW[:,sim_factors[comp]]
            flip.append(1)
        elif corr[sim_factors[comp],comp] < 0:
            #flip sign of the factor
            W[:,comp] =  - tempW[:,sim_factors[comp]]
            flip.append(-1)
    return W, sim_factors, flip, maxcorr                        

def plot_loadings(W, d, W_path):

    """ 
    Plot loadings.

    Parameters
    ----------
    W : array-like, shape(n_features, n_comps)
        Concatenated loading matrix. The number of features
        here correspond to the total number of features in 
        all groups. 

    d : list
        Number of features in each group.

    W_path : str
        Path to save the figure.     
    
    """
    x = np.linspace(1,3,num=W.shape[0])
    step = 6
    c = step * W.shape[1]+1
    plt.figure()
    for col in range(W.shape[1]):
        y = W[:,col] + (col+c)
        c-=step
        plt.plot( x, y, color='black', linewidth=1.5)
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    s = 0
    for j in range(len(d)-1):
        plt.axvline(x=x[d[j]+s-1],color='red')
        s += d[j]+1  
    plt.savefig(W_path)
    plt.close()

def plot_Z(Z, Z_path, match=False, flip=None):

    """ 
    Plot latent variables.

    Parameters
    ----------
    Z : array-like, shape(n_features, n_comps)
        Latent variables matrix.

    Z_path : str
        Path to save the figure.

    match : bool, defaults to False.
        Match (or not) the latent factors.

    flip : list, defaults to None.
        Indices to flip the latent factors. Positive correlation 
        corresponds to the same sign and negative 
        correlation represents inverse sign.    
    
    """   
    x = np.linspace(0, Z.shape[0], Z.shape[0])
    ncomp = Z.shape[1]
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j+1)    
        if match:
            ax.scatter(x, Z[:, j] * flip[j], s=4)
        else:
            ax.scatter(x, Z[:, j], s=4)
        ax.set_xticks([])
        ax.set_yticks([])       
    plt.savefig(Z_path)
    plt.close()           

def plot_params(model, res_dir, args, best_run, data, plot_trueparams=False, plot_medianparams=False):
    
    """ 
    Plot the model parameters and ELBO.

    Parameters
    ----------
    model : Outputs of the model.

    res_dir : str
        Path to the directory where the results will be saved.   

    args : local namespace 
        Arguments selected to run the model.

    best_run : int
        Index of the best model.

    data : dict
        Training and test data, as well as the model parameters 
        used to generate the data.

    plot_trueparams : bool, defaults to False.
        Plot (or not) the model parameters used to generate the 
        data.                
    
    plot_medianparams : bool, defaults to False.
        Plot (or not) the output model parameters when the missing
        values were imputed using the median before training the 
        model. 

    """
    file_ext = '.png' #file extension to save the plots
    #Concatenate loadings and alphas across groups    
    W_est = np.zeros((np.sum(model.d),model.k))
    alphas_est = np.zeros((model.k, args.num_groups))
    W_true = np.zeros((np.sum(model.d),data['true_K']))
    if plot_trueparams:
        alphas_true = np.zeros((data['true_K'], args.num_groups))
    d = 0
    for m in range(args.num_groups):
        Dm = model.d[m]
        if plot_trueparams:
            alphas_true[:,m] = data['alpha'][m]
        W_true[d:d+Dm,:] = data['W'][m]
        alphas_est[:,m] = model.E_alpha[m]
        W_est[d:d+Dm,:] = model.means_w[m]
        d += Dm  

    # Plot loading matrices
    if plot_trueparams:
        #plot true Ws
        W_path = f'{res_dir}/[{best_run+1}]W_true{file_ext}'
        plot_loadings(W_true, model.d, W_path) 
    
    #plot inferred Ws
    if model.k == data['true_K']:
        #match true and inferred factors
        match_res = match_factors(W_est, W_true)
        W_est = match_res[0] 
    if plot_medianparams:                          
        W_path = f'{res_dir}/[{best_run+1}]W_est_median{file_ext}'
    else:
        W_path = f'{res_dir}/[{best_run+1}]W_est{file_ext}'           
    plot_loadings(W_est, model.d, W_path)
    
    # Plot latent variables
    if plot_trueparams:    
        #plot true latent variables 
        Z_path = f'{res_dir}/[{best_run+1}]Z_true{file_ext}'    
        plot_Z(data['Z'], Z_path)
    #plot inferred latent variables
    if plot_medianparams:                          
        Z_path = f'{res_dir}/[{best_run+1}]Z_est_median{file_ext}'
    else:
        Z_path = f'{res_dir}/[{best_run+1}]Z_est{file_ext}'    
    if model.k == data['true_K']:
        simcomps = match_res[1]
        plot_Z(model.means_z[:, simcomps], Z_path, match=True, flip=match_res[2])
    else:     
        plot_Z(model.means_z, Z_path)       

    # Plot alphas
    if plot_trueparams:
        #plot true alphas
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_true{file_ext}'
        hinton_diag(np.negative(alphas_true.T), alphas_path)     
    #plot inferred alphas
    if plot_medianparams:                          
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est_median{file_ext}'
    else:
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est{file_ext}'
    if model.k == data['true_K']:
        hinton_diag(np.negative(alphas_est[simcomps,:].T), alphas_path) 
    else:
        hinton_diag(np.negative(alphas_est.T), alphas_path)

    # Plot ELBO
    if plot_medianparams:
        L_path = f'{res_dir}/[{best_run+1}]ELBO_median{file_ext}'
    else:
        L_path = f'{res_dir}/[{best_run+1}]ELBO{file_ext}'    
    plt.figure(figsize=(5, 4))
    plt.plot(model.L[1:])
    plt.savefig(L_path)
    plt.close() 


       

        
   