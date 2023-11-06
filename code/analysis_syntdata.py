""" Run experiments on simulated data """

import os
import numpy as np
import time
import pickle
import os
import copy
import argparse
import visualization_syntdata
from models import GFA_DiagonalNoiseModel, GFA_OriginalModel
from utils import GFAtools

def generate_data(res_dir, run, args, missing_info):
    """
    Load data from a file if it exists or generate new data and save it to a file.

    Parameters:
    - res_dir: Directory where the data file is stored or will be saved.
    - run: The current iteration of the model execution.
    - args: Arguments selected to run the model.
    - missing_info: Information about missing data patterns.

    Returns:
    - simulated_data: A dictionary containing the generated data.
    """

    # File to save or load data
    data_file = os.path.join(res_dir, f'[{run + 1}]Data.dictionary')

    # Check for existing data or generate new data
    if os.path.exists(data_file):
        with open(data_file, 'rb') as file:
            simulated_data = pickle.load(file)
        print("Loaded previously generated data.")
    else:
        simulated_data = generate_parameters(args, args.num_groups, missing_info if args.scenario != 'complete' else None)
        with open(data_file, 'wb') as file:
            pickle.dump(simulated_data, file)
        print("Generated new data and saved to file.")
    
    return simulated_data

def generate_d(num_groups):
    """
    Generate a random number of features for each group.

    Parameters:
    - num_groups: The number of groups for which to generate feature numbers.

    Returns:
    - A numpy array containing the number of features for each group.
    """

    rng = np.random.default_rng(num_groups)
    return rng.choice([50, 40, 30], num_groups, replace=True)

def generate_tau(num_groups, d):
    """
    Generate tau values (precision parameters) for each feature within each group.

    Parameters:
    - num_groups: The number of groups for which to generate tau values.
    - d: An array or list containing the number of features for each group.

    Returns:
    - tau: A list of arrays, where each array contains the tau values for a group's features.
    """

    rng = np.random.default_rng(num_groups)
    tau = []
    for i in range(num_groups):
        tau_precision = rng.integers(3, 11)
        tau.append(tau_precision * np.ones(int(d[i])))
    return tau

def generate_missing_data(X_train, missing_info):
    """
    Generate missing entries in the training datasets for simulation purposes.
    This simulates scenarios where data is not fully observed, by introducing NaNs based
    on the percentage specified in missing_info.

    Parameters
    ----------
    X_train : list of np.array
        List of data matrices representing training data for each group before introducing missing values.
    missing_info : dict
        Dictionary specifying the percentage of missing values ('perc') and the datasets ('ds') 
        which should have missing values introduced.

    Returns
    -------
    X_train : list of np.array
        The input training data matrices after introducing NaN values to simulate missing data.
    missing_Xtrue : list of np.array
        Matrices of the same shape as X_train groups containing the original data values that have been
        replaced with NaNs in the X_train datasets.
    """
    missing_Xtrue = []
    
    for i, g_miss in enumerate(missing_info['ds']):
        group_idx = g_miss - 1
        group = X_train[group_idx]
        
        # Probability of a value being missing
        p_missing = missing_info['perc'][i] / 100

        # Randomly decide which entries are missing
        missing_mask = np.random.rand(*group.shape) < p_missing
        
        # Save the true missing values
        missing_Xtrue.append(group * missing_mask)
        
        # Replace missing entries with NaN
        group[missing_mask] = np.nan
        X_train[group_idx] = group
    
    return X_train, missing_Xtrue



def generate_parameters(args, num_groups, missing_info=None):
    """
    Generate parameters for the simulation of data according to the specified arguments.
    This function defines the latent factors, noise precision, and the sharing structure of the factors.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    num_groups : int
        Number of data groups for which the parameters should be generated.
    missing_info : dict, optional
        Dictionary specifying parameters for generating missing data.

    Returns
    -------
    data : dict
        Dictionary containing the generated parameters, including training and test data, 
        factor loading matrices, precision of the noise, and other relevant model parameters.
    """
    
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest  # total number of samples
    M = num_groups  # number of groups
    d = generate_d(M)  # number of variables in each group
    true_K = 4  # Number of latent factors

    # Initialize Z with zeros
    Z = np.zeros((N, true_K))

    # Specify Z manually
    Z = np.zeros((N, true_K))
    for i in range(0, N):
        Z[i,0] = np.sin((i+1)/(N/20))
        Z[i,1] = np.cos((i+1)/(N/20))
        Z[i,2] = 2 * ((i+1)/N-0.5)    
    Z[:,3] = np.random.normal(0, 1, N)   
    
    # Initialize alpha to inactive
    alpha = np.ones((M, true_K)) * 1e6
    rng = np.random.default_rng(num_groups)
    
    # Dictionary to store information about factor sharing
    factor_sharing_info = {
        'shared_all_groups': [],
        'shared_subgroups': {},
        'specific_group': {}
    }
    # Set shared factors active for all groups
    alpha[:, 0] = 1
    factor_sharing_info['shared_all_groups'].append(1)
    
    alpha[0, 1] = 1
    factor_sharing_info['specific_group'][2] = 1
    
    # Set shared factors between subgroups
    for i in range(2, true_K):
        subgroup_indices = rng.choice(range(1, M), size=M // 2, replace=False)
        alpha[subgroup_indices, i] = 1
        print(f"Latent Factor {i + 1} is assign to group {subgroup_indices + 1}")
   
    tau = generate_tau(num_groups, d)
    W = [[] for _ in range(num_groups)]
    X_train = [[] for _ in range(num_groups)]
    X_test = [[] for _ in range(num_groups)]
    
    for i in range(num_groups):
        W[i] = np.random.normal(0, 1/np.sqrt(alpha[i, :]), (d[i], true_K))
        X = Z @ W[i].T + np.random.normal(0, 1/np.sqrt(tau[i]), (N, d[i]))
        X_train[i] = X[:Ntrain, :]
        X_test[i] = X[Ntrain:, :]
        
        
    # latent variables for training the model    
    Z = Z[0:Ntrain, :]
    
    # Nested loop to print every value of alpha
    for i in range(num_groups):
        print(f"Group {i + 1}:")
        for j in range(true_K):
            print(f"Factor {j + 1}: {alpha[i, j]}")
        print()  # Print a blank line between groups
    
    
    # Print information
    print("Number of variables and tau precision in each group:")
    for i, (num_variables, tau_values) in enumerate(zip(d, tau), start=1):
        print(f"Group {i}: {num_variables} variables, Tau precision: {tau_values[0]}")

    # Generate incomplete training data
    if args.scenario == 'incomplete':
        X_train, missing_Xtrue = generate_missing_data(X_train, missing_info)

    # Store data and model parameters
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        # Save true missing values
        data.update({'trueX_miss': missing_Xtrue})
    return data

# def compute_MSE(simulated_data, model, args):
#     """
#     Calculate the Mean Squared Error (MSE) of the model's predictions. This function compares the last group's
#     test data (treated as unobserved during training) with the predictions made by the model to evaluate its performance.

#     Parameters
#     ----------
#     simulated_data : dict
#         A dictionary containing the simulated training and test data ('X_tr' and 'X_te').
#     model : GFA model object
#         The trained Group Factor Analysis model that will be used to predict missing data.
#     args : local namespace
#         Arguments selected to run the model.

#     Returns
#     -------
#     MSE : float
#         The mean squared error between the true test data and the predicted data for the unobserved group.
#     """
#     obs_ds = np.ones(args.num_groups, dtype=int)
#     obs_ds[-1] = 0  # Set the last group as unobserved
#     X_test = simulated_data['X_te']
#     X_pred = GFAtools(X_test, model).PredictGroups(obs_ds, args.noise)
#     return np.mean((X_test[-1] - X_pred[0]) ** 2)     



def compute_MSE(simulated_data, model, args):
    """
    Calculate the Mean Squared Error (MSE) of the model's predictions. This function compares the first group's
    test data (treated as unobserved during training) with the predictions made by the model to evaluate its performance.

    Parameters
    ----------
    simulated_data : dict
        A dictionary containing the simulated training and test data ('X_tr' and 'X_te').
    model : GFA model object
        The trained Group Factor Analysis model that will be used to predict missing data.
    args : local namespace
        Arguments selected to run the model.

    Returns
    -------
    MSE : float
        The mean squared error between the true test data and the predicted data for the unobserved group.
    """
    # Initialize an array to mark all groups as observed
    obs_ds = np.ones(args.num_groups, dtype=int)
    # Set the first group as unobserved
    obs_ds[0] = 0  
    # Retrieve the test data for all groups
    X_test = simulated_data['X_te']
    # Use the model to predict the unobserved group's data
    X_pred = GFAtools(X_test, model).PredictGroups(obs_ds, args.noise)
    # Compute the MSE between the unobserved group's true test data and the predicted data
    MSE = np.mean((X_test[0] - X_pred[0]) ** 2)
    
    return MSE

def compute_chance_level_MSE(GFAmodel, simulated_data):
    """
    Calculate and set the chance level MSE for the Group Factor Analysis model by comparing the test set
    of the unobserved group with the mean of the training set.

    Parameters
    ----------
    GFAmodel : object
        The trained Group Factor Analysis model object.
    simulated_data : dict
        A dictionary containing the simulated dataset including the training and test sets.
    """
    unobserved_group_means = np.nanmean(simulated_data['X_tr'][-1], axis=0)
    chance_level_means = np.tile(unobserved_group_means, (simulated_data['X_te'][-1].shape[0], 1))
    GFAmodel.MSE_chlev = np.nanmean((simulated_data['X_te'][-1] - chance_level_means) ** 2)

def compute_correlation(GFAmodel, simulated_data, missing_info):
    """
    Compute the correlation between the true missing values and the predicted values by the GFA model.

    Parameters
    ----------
    GFAmodel : object
        The GFA model object after training.
    simulated_data : dict
        A dictionary containing the simulated data with true missing values.
    missing_info : dict
        A dictionary with information about the groups and percentages of missing data.

    Updates
    -------
    GFAmodel.Corr_miss : np.ndarray
        An array of correlation coefficients for each group with missing data, updated in the GFA model object.
    """
    Corr_miss = np.zeros(len(missing_info['ds']))
    missing_pred = GFAtools(simulated_data['X_tr'], GFAmodel).PredictMissing(missing_info)
    missing_true = simulated_data['trueX_miss']
    for idx, group in enumerate(missing_info['ds']):
        true_vals = missing_true[idx].flatten()
        pred_vals = missing_pred[idx].flatten()
        Corr_miss[idx] = np.corrcoef(true_vals[true_vals != 0], pred_vals[~np.isnan(pred_vals)])[0, 1]
    GFAmodel.Corr_miss = Corr_miss.reshape(1, -1)

def run_model(simulated_data, args, missing_info, run, res_dir):
    """
    Run the GFA model, compute performance metrics, and save the model to a file.

    This function handles the entire process of initializing the GFA model, fitting it to the data,
    computing MSE and correlation metrics, and saving the model results for later analysis.

    Parameters
    ----------
    simulated_data : dict
        A dictionary containing the simulated training and test data.
    args : local namespace
        Arguments selected to run the model.
    missing_info : dict
        A dictionary with information about the groups and percentages of missing data, if applicable.
    run : int
        The current iteration of the model execution.
    res_dir : str
        The directory where the model results will be saved.

    """
    res_file = os.path.join(res_dir, f'[{run + 1}]ModelOutput.dictionary')
    if not os.path.exists(res_file):
        print("Running the model...")
        X_train = simulated_data['X_tr']
        params = {'num_groups': args.num_groups, 'K': args.K, 'scenario': args.scenario}
        
        # Select the model based on noise type
        GFAmodel = GFA_DiagonalNoiseModel(X_train, params) if 'diagonal' in args.noise else GFA_OriginalModel(X_train, params)
        assert 'diagonal' in args.noise or params['scenario'] == 'complete', "For non-diagonal noise, the scenario must be complete."

        # Fit the model and time the process
        start_time = time.process_time()
        GFAmodel.fit(X_train)
        elapsed_time = time.process_time() - start_time
        print(f'Computational time: {elapsed_time:.2f}s')

        # Compute and store MSE for predictions
        GFAmodel.MSE = compute_MSE(simulated_data, GFAmodel, args)

        # Calculate the MSE at chance level
        compute_chance_level_MSE(GFAmodel, simulated_data)

        # Predict missing values and compute correlation coefficients
        if args.scenario == 'incomplete':
            compute_correlation(GFAmodel, simulated_data, missing_info)

        # Save the GFA model to a file
        with open(res_file, 'wb') as file:
            pickle.dump(GFAmodel, file)
    else:
        print('Model already computed before and saved in this path.')

def impute_median(X_train, missing_info):
    """
    Impute the missing values in the training data with the median of observed values in each feature.

    Parameters
    ----------
    X_train : list of np.array
        The input list containing training data matrices for each group with missing values.
    missing_info : dict
        Dictionary with 'ds' key indicating the indices of groups with missing values.

    Returns
    -------
    X_imputed : list of np.array
        The list containing training data matrices with missing values imputed by the median of each feature.
    """
    X_imputed = copy.deepcopy(X_train)
    for g in (np.array(missing_info['ds']) - 1):
        for j in range(X_imputed[g].shape[1]):
            valid_data = X_imputed[g][:,j][~np.isnan(X_imputed[g][:,j])]
            median_value = np.nanmedian(valid_data)
            X_imputed[g][:,j] = np.where(np.isnan(X_imputed[g][:,j]), median_value, X_imputed[g][:,j])
    return X_imputed

def run_model_with_imputation(simulated_data, missing_info, args, run, res_dir):
    """
    Runs the Group Factor Analysis (GFA) model on data where missing values have been imputed with the median.
    Saves the model output and performance metrics.

    Parameters
    ----------
    simulated_data : dict
        Dictionary containing 'X_tr' key with training data.
    missing_info : dict
        Dictionary containing information about which datasets have missing values.
    args : local namespace
        Arguments and settings for the model execution.
    run : int
        The current run number for the experiment.
    res_dir : str
        Directory path where model results should be saved.

    """
    # Impute missing values with median
    X_imputed = impute_median(simulated_data['X_tr'], missing_info)

    # Prepare file path
    res_med_file = os.path.join(res_dir, f'[{run+1}]ModelOutput_median.dictionary')
    if os.path.exists(res_med_file):
        print("Model output after median imputation already exists.")

    print("Running model after imputation with median...")
    params = {'num_groups': args.num_groups, 'K': args.K, 'scenario': args.scenario}

    # Select the appropriate model
    GFAmodel_median = GFA_DiagonalNoiseModel(X_imputed, params, imputation=True) if 'diagonal' in args.noise else GFA_OriginalModel(X_imputed, params)

    # Fit the model and record computational time
    start_time = time.process_time()
    GFAmodel_median.fit(X_imputed)
    computational_time = time.process_time() - start_time
    print(f'Computational time: {computational_time:.2f}s')

    # Compute and store MSE for predictions
    GFAmodel_median.MSE = compute_MSE(simulated_data, GFAmodel_median, args)

    # Save the model to a file
    with open(res_med_file, 'wb') as parameters:
        pickle.dump(GFAmodel_median, parameters)


def main(args):
    
    """ 
    Run experiments on simulated data based on the input args.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    
    """
    # Define parameters to generate incomplete data sets
    missing_info = {'perc': [60], 'ds': [1]} if args.scenario == 'incomplete' else {}

    # Construct directory path and create directory if it does not exist
    flag = f's{missing_info.get("ds", "")}_{missing_info.get("perc", "")}/' if missing_info else ''
    res_dir = os.path.join('results1111', 'final', f'{args.num_groups}groups', f'GFA_{args.noise}', f'{args.K}comps', args.scenario, flag)
    os.makedirs(res_dir, exist_ok=True)
    
    # Run model num_runs times       
    for run in range(args.num_runs):
        print(f'------------\nRun: {run + 1}')
        simulated_data = generate_data(res_dir, run, args, missing_info)   

        # Run model
        run_model(simulated_data, args, missing_info, run, res_dir)
        
        # Run model with median imputation approach if impMedian=True
        if args.impMedian:
            run_model_with_imputation(simulated_data, missing_info, args, run, res_dir)              

    # Plot and save results
    print('Plotting results...')
    visualization_syntdata.get_results(args, res_dir, missing_info=missing_info if 'incomplete' in args.scenario else None)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", nargs='?', default='incomplete', type=str,
                        help='Data scenario (complete or incomplete)')
    parser.add_argument("--noise", nargs='?', default='diagonal', type=str,
                        help='Noise assumption for GFA models (diagonal or spherical)')
    parser.add_argument("--num-groups", nargs='?', default=2, type=int,
                        help='Number of groups')
    parser.add_argument("--K", nargs='?', default=15, type=int,
                        help='number of factors to initialise the model')
    parser.add_argument("--num-runs", nargs='?', default=10, type=int,
                        help='number of random initializations (runs)')
    parser.add_argument("--impMedian", nargs='?', default=True, type=bool,
                        help='(not) impute median')
    args = parser.parse_args()

    main(args)    


