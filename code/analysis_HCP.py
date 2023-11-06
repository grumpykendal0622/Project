""" Run experiments on HCP data """

import argparse
import time
import os
import pickle
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import visualization_HCP 
from scipy import io
from sklearn.preprocessing import StandardScaler
from models import GFA_DiagonalNoiseModel, GFA_OriginalModel
from utils import GFAtools

def split_data(ptrain, num_subjects):
    """Split data into training and test sets."""
    np.random.shuffle(num_subjects)
    train_idx = num_subjects[:int(ptrain * len(num_subjects))]
    test_idx = num_subjects[int(ptrain * len(num_subjects)):]
    return train_idx, test_idx

def standardize_data(X_train, X_test):
    """Standardize training and test data."""
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)
    return X_train_standardized, X_test_standardized

def remove_missing_values(X_train, group_idx, p_miss):
    """
    Remove values randomly from a specified group to simulate missing data.
    
    Parameters
    ----------
    X_train : list of np.array
        List of training data for each group.
    group_idx : int
        Index of the group from which missing values should be removed.
    p_miss : float
        Percentage of missing values to be introduced.

    Returns
    -------
    X_train : list of np.array
        Training data after introducing missing values.
    missing_trueX : np.array
        The original values that have been set to missing.
    """
    # Calculate the probability of missing
    prob_miss = p_miss / 100

    # Choose values to be missing
    missing = np.random.rand(*X_train[group_idx].shape) < prob_miss

    # Save the true missing values
    missing_trueX = np.where(missing, X_train[group_idx], 0)

    # Introduce missing values
    X_train[group_idx][missing] = np.nan

    return X_train, missing_trueX

def create_gfa_model(X_train, args):
    """
    Create and initialize the Group Factor Analysis (GFA) model.
    
    Parameters:
    - X_train: The training data set.
    - args: The argument namespace containing model parameters.

    Returns:
    - An instance of the GFA model, initialized with the provided data and parameters.
    """
    params = {'num_groups': args.num_groups, 'K': args.K, 'scenario': args.scenario}
    if 'diagonal' in args.noise:
        model = GFA_DiagonalNoiseModel(X_train, params)
    else:
        model = GFA_OriginalModel(X_train, params)
    return model

def compute_MSE(X_train, X_test, model):
    """
    Calculate Mean Squared Errors (MSE) for non-imaging (NI) measures.

    Parameters
    ----------
    X_train : list of numpy.ndarray
        Training data for all groups; each array is for one group.
    
    X_test : list of numpy.ndarray
        Test data for all groups; each array is for one group.
    
    model : model object
        The trained model that provides predictions.

    Returns
    -------
    MSE_NI_test : numpy.ndarray
        The MSEs for each NI measure on the test dataset.
    
    MSE_NI_train_mean : numpy.ndarray
        The MSEs for each NI measure comparing test data to the training mean.
    """
    # group 2 is for non-imaging measures
    non_imaging_group_index = 1
    
    # Calculate the mean of non-imaging measures across training data
    NI_train_mean = np.nanmean(X_train[non_imaging_group_index], axis=0)
    
    # Identify which group was not observed (group 2 in this case)
    observed_indicator = np.array([1, 0])  # 1 indicates observed group
    group_to_predict_index = np.where(observed_indicator == 0)[0][0]
    
    # Use the model to predict the non-observed group data
    X_predicted = GFAtools(X_test, model).PredictGroups(observed_indicator, args.noise)
    
    # Initialize MSE arrays for test predictions and training mean comparisons
    MSE_NI_test = np.zeros(model.d[non_imaging_group_index])
    MSE_NI_train_mean = np.zeros(model.d[non_imaging_group_index])
    
    # Calculate MSE for each non-imaging measure
    for j in range(model.d[non_imaging_group_index]):
        # MSE between test data and predictions
        MSE_NI_test[j] = np.mean((X_test[group_to_predict_index][:, j] - X_predicted[0][:, j]) ** 2)
        
        # MSE between test data and training mean
        MSE_NI_train_mean[j] = np.mean((X_test[group_to_predict_index][:, j] - NI_train_mean[j]) ** 2)
    
    # Normalize MSE by the variance of the test data
    MSE_NI_test /= np.var(X_test[group_to_predict_index], axis=0)
    MSE_NI_train_mean /= np.var(X_test[group_to_predict_index], axis=0)

    return MSE_NI_test, MSE_NI_train_mean


def create_result_directory(base_directory, experiment_arguments):
    """
    Create a directory to save experiment results, based on the experiment settings.
    
    Parameters
    ----------
    base_directory : str
        The base directory where the experiments directory will be created.
    
    experiment_arguments : Namespace
        A namespace or an object that holds the arguments for the experiment settings.
    
    Returns
    -------
    str
        The path to the created results directory.
    """
    experiments_dir = os.path.join(base_directory, 'final')
    
    # Determine the directory flag based on whether the experiment is 'complete' or with 'missing' data
    if experiment_arguments.scenario == 'complete':
        directory_flag = f'training{experiment_arguments.ptrain}'
    else:
        directory_flag = f's{experiment_arguments.gmiss}_{experiment_arguments.tmiss}{experiment_arguments.pmiss}_training{experiment_arguments.ptrain}'
    
    # Create the results directory path
    results_dir = os.path.join(experiments_dir, f'GFA_{experiment_arguments.noise}', f'K{experiment_arguments.K}', experiment_arguments.scenario, directory_flag)
    
    # Create the directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return results_dir

def load_data(data_directory, num_groups):
    """
    Load preprocessed brain and clinical data matrices.
    
    Parameters
    ----------
    data_directory : str
        The directory where the data files are located.
    
    num_groups : int
        The number of groups in the dataset.
    
    Returns
    -------
    list of numpy.ndarray
        A list containing data matrices for each group.
    """
    # Load brain data and clinical data matrices from .mat files
    brain_data = io.loadmat(os.path.join(data_directory, 'X.mat'))
    clinical_data = io.loadmat(os.path.join(data_directory, 'Y.mat'))
    
    # Load labels for clinical data from an Excel file
    labels_df = pd.read_excel(os.path.join(data_directory, 'LabelsY.xlsx'))
    y_labels = labels_df['Label'].values
    
    # Initialize a list to hold the data for each group
    data_groups = [None] * num_groups
    data_groups[0] = brain_data['X']
    data_groups[1] = clinical_data['Y']
    
    return data_groups, y_labels



def main(args): 

    """ 
    Main function to run experiments on HCP data.

    Parameters
    ----------
    args : local namespace 
        Input arguments selected to run the model.
    
    """

    res_dir = create_result_directory(args.dir, args)
    X , Y_labels = load_data(args.dir, args.num_groups)


    for run in range(0, args.num_runs):
        print("Run: ", run+1)
        filepath = f'{res_dir}[{run+1}]Results.dictionary'
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as parameters:
                pickle.dump(0, parameters)
                
            N = X[0].shape[0]
            all_subjects = np.arange(N)
            
            train_indices, test_indices = split_data(args.ptrain / 100, all_subjects)
            
            X_train = [group[train_indices, :] for group in X]
            X_test = [group[test_indices, :] for group in X]
            
            if args.scenario == 'complete':
                X_train, X_test = map(list, zip(*(standardize_data(group_train, group_test) for group_train, group_test in zip(X_train, X_test))))
            
            assert round((len(train_indices) / N) * 100) == args.ptrain
            
            if args.scenario == 'incomplete':
                X_train, missing_trueX = remove_missing_values(X_train, args.gmiss, args.pmiss, args.tmiss)
                X_train, X_test = map(list, zip(*(standardize_data(group_train, group_test) for group_train, group_test in zip(X_train, X_test))))
            
            GFAmodel = create_gfa_model(X_train, args)
             
            
             # Running the GFA model
            print("Start running model---------")
            time_start = time.process_time()

            # Fit the GFA model with the training data
            GFAmodel.fit(X_train)

            # Calculate the elapsed time for the model fitting
            GFAmodel.time_elapsed = time.process_time() - time_start
            # print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed/60))} min')

            # Compute the mean squared error (MSE) for training and test data
            GFAmodel.MSEs_NI_te, GFAmodel.MSEs_NI_tr = compute_MSE(X_train, X_test, GFAmodel)

            # If the data is incomplete, predict the missing values
            if args.scenario == 'incomplete':
                # Define the information about missing data
                missing_info = {
                    'perc': [args.pmiss], # percentage of missing data
                    'type': [args.tmiss], # type of missing data
                    'ds': [args.gmiss]    # groups with missing values
                }
                
                # Predict missing values using the GFA model
                miss_pred = GFAtools(X_train, GFAmodel).PredictMissing(missing_info)
                
                # Calculate the correlation between true missing values and predicted values
                GFAmodel.corrmiss = np.corrcoef(
                    missing_trueX[missing_trueX != 0], 
                    miss_pred[0][np.logical_not(np.isnan(miss_pred[0]))]
                )[0, 1]
                
                # Cleanup: Remove mask with NaNs from the model output dictionary
                if hasattr(GFAmodel, 'X_nan'):
                    del GFAmodel.X_nan

            # Save the model outputs to a file for later use
            with open(filepath, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)
            

    # get plots 
    print('Plotting and saving results--------')
    visualization_HCP.get_results(args, Y_labels, res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GFA on HCP data")
    parser.add_argument('--dir', type=str, default='/Users/yangbowen/Desktop/192/data',
                        help='Project directory')                   
    parser.add_argument('--noise', type=str, default='diagonal', 
                        help='Noise assumption for GFA models (diagonal or spherical)') 
    parser.add_argument('--num_groups', type=int, default=2, 
                        help='Number of groups')                                                          
    parser.add_argument('--K', type=int, default=80,
                        help='number of factors to initialise the model')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='number of random initializations (runs)')
    parser.add_argument('--ptrain', type=int, default=80,
                        help='Percentage of training data')
    parser.add_argument('--scenario', type=str, default='complete',
                        help='Data scenario (complete or incomplete)')                                        
    # Missing data info
    # (This is only needed if one wants to simulate how the model handles and
    # predicts missing data)
    # parser.add_argument('--pmiss', type=int, default=20,
    #                     help='Percentage of missing data')
    # parser.add_argument('--tmiss', type=str, default='rows',
    #                     help='Type of missing data (random values or rows)')
    # parser.add_argument('--gmiss', type=int, default=1,
    #                     help='Group with missing data')
    args = parser.parse_args()

    main(args) 

