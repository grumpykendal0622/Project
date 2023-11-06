""" Plot and save the results of the experiments on HCP data """

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from scipy import io


def plot_MSE(res_path, MSE_NI_test, MSE_NI_train, y_labels):
    """
    Plot Mean Squared Error (MSE) of each non-imaging subject measure for both test and training sets.

    Parameters:
    - res_path: str, path to the directory where the plot will be saved.
    - MSE_NI_test: numpy.ndarray, array containing the MSE for the test set.
    - MSE_NI_train: numpy.ndarray, array containing the MSE for the training set.
    - y_labels: list or numpy.ndarray, labels for the non-imaging subject measures.

    Returns:
    - None
    """
    plt.figure(figsize=(10,6))
    pred_path = f'{res_path}/Predictions.png'
    x = np.arange(len(y_labels))  # Changed from MSE_NI_test.shape[1] for clarity
    plt.errorbar(x, np.mean(MSE_NI_test, axis=0), yerr=np.std(MSE_NI_test, axis=0), fmt='bo', label='Test Predictions')
    plt.errorbar(x, np.mean(MSE_NI_train, axis=0), yerr=np.std(MSE_NI_train, axis=0), fmt='yo', label='Training Mean')
    plt.legend(loc='upper left', fontsize=17)
    plt.ylim((np.min(MSE_NI_test) - 0.2, np.max(MSE_NI_test) + 0.1))
    plt.title('Prediction of NI measures from brain connectivity', fontsize=22)
    plt.xlabel('Non-imaging subject measures', fontsize=19)
    plt.ylabel('Relative MSE', fontsize=19)
    plt.xticks(x, y_labels, rotation=90, fontsize=14)  # Include y_labels for x-axis
    plt.yticks(fontsize=14)
    # plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(pred_path)
    plt.close()
    

def calculate_variance_explained(GFA_output, args):
    """
    Calculate the total variance explained by the factors and the variance explained by noise.
    
    Parameters:
    - GFA_output: object containing the output of the GFA model, expected to have
                  attributes `s`, `means_w`, and `E_tau`.
    - args: object containing the runtime arguments, expected to have an attribute
            'noise' which can contain the string 'spherical' indicating the noise model.
    
    Returns:
    - total_variance: float, the total variance explained by the model including noise.
    - factors_variance: float, the variance explained by the factors alone.
    """
   
    total_variance = 0
    factors_variance = 0
    for s in range(GFA_output.s):
        w = GFA_output.means_w[s]
        noise_term = 'spherical' in args.noise
        T = (1 / GFA_output.E_tau[0, s]) * np.identity(w.shape[0]) if noise_term else np.diag(1 / GFA_output.E_tau[s][0, :])
        total_variance += np.trace(np.dot(w, w.T) + T)
        factors_variance += np.trace(np.dot(w, w.T))
    return total_variance, factors_variance


def get_factors(model, res_dir, BestModel=False):
    """
    Find the most relevant shared and specific factors from the GFA model output.

    Parameters
    ----------
    model : object
        The trained model containing the weights and explained variance.

    res_dir : str
        The directory where the results will be saved.

    BestModel : bool, optional
        Flag indicating whether to save the results for the best model only.
        Defaults to False.

    Returns
    -------
    tuple of (relfactors_shared, relfactors_specific)
        relfactors_shared : list
            List of indices of shared relevant factors across groups.
        relfactors_specific : list of lists
            List of lists where each sublist contains indices of relevant factors specific to a group.
    """
    # Concatenate weights from all groups and calculate total variance
    W = np.concatenate((model.means_w[0], model.means_w[1]), axis=0)
    ncomps = W.shape[1]
    total_var = model.VarExp_total
    
    # Initialize variables for within-group variance
    var_within = np.zeros((model.s, ncomps))
    d = 0

    # Calculate explained variance for each factor within groups
    for s in range(model.s):
        Dm = model.d[s]
        var_within[s, :] = (np.sum(W[d:d+Dm, :] ** 2, axis=0) / total_var) * 100
        d += Dm
    
    # Calculate relative explained variance for each factor within groups
    relvar_within = (var_within.T / np.sum(var_within, axis=1)).T * 100

    # Identify shared and specific relevant factors
    relfactors_shared = []
    relfactors_specific = [[] for _ in range(model.s)]
    ratio = var_within[1, :] / var_within[0, :]

    # Thresholds for determining relevance
    RELVANCE_THRESHOLD = 7.5
    RATIO_UPPER_BOUND = 300
    RATIO_LOWER_BOUND = 0.001

    # Assess each factor for its relevance and specificity
    for c in range(ncomps):
        if np.any(relvar_within[:, c] > RELVANCE_THRESHOLD):
            if ratio[c] > RATIO_UPPER_BOUND:
                relfactors_specific[1].append(c)
            elif ratio[c] < RATIO_LOWER_BOUND:
                relfactors_specific[0].append(c)
            else:
                relfactors_shared.append(c)
    
    # If the best model is being processed, save variance information to an Excel file
    if BestModel:
        var_path = f'{res_dir}/Info_factors.xlsx'
        df = pd.DataFrame({
            'Factors': np.arange(1, ncomps + 1),
            'Relvar (brain)': relvar_within[0, :],
            'Relvar (NI measures)': relvar_within[1, :],
            'Var (brain)': var_within[0, :],
            'Var (NI measures)': var_within[1, :],
            'Ratio (NI/brain)': ratio
        })

        df.to_excel(var_path, index=False)

    return relfactors_shared, relfactors_specific


def get_results(args, y_labels, res_path):

    """ 
    Iterate through each run to process and log the results.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    
    y_labels : array-like
        Array of strings with the labels of the non-imaging
        subject measures.

    res_dir : str
        Path to the directory where the results will be 
        saved.       
    
    """
    num_runs = args.num_runs #number of runs
    #initialise variables to save MSEs, correlations and ELBO values
    MSE_NI_test = np.zeros((num_runs, y_labels.size))
    MSE_NI_train = np.zeros((num_runs, y_labels.size))
    Corr_miss = np.zeros((1,num_runs)) if args.scenario == 'incomplete' else None
    ELBO = np.zeros((1,num_runs))
    #initialise file where the results will be written
    ofile = open(f'{res_path}/results.txt','w')   

    # Iterate through each run to process and log the results.
    for i in range(num_runs):
        print(f'\nRun: {i + 1}', file=ofile)
        print('-' * 48, file=ofile)  # Create a separator line.

        # Construct the file path to load the results from.
        result_dict_file = f'{res_path}[{i+1}]Results.dictionary'

        # Check if the results file exists and is not empty.
        if os.path.isfile(result_dict_file) and os.stat(result_dict_file).st_size > 5:
            # Load the model output from the dictionary file.
            with open(result_dict_file, 'rb') as parameter_file:
                GFA_output = pickle.load(parameter_file)

            # Log the computational time and number of factors estimated.
            computational_time = np.round(GFA_output.time_elapsed / 60, 2)  # Convert seconds to minutes.
            print(f'Computational time (minutes): {computational_time}', file=ofile)
            print(f'Total number of factors estimated: {GFA_output.k}', file=ofile)

            # Record and log the last value of ELBO.
            ELBO[0, i] = GFA_output.L[-1]
            print(f'ELBO (last value): {np.around(ELBO[0, i], 2)}', file=ofile)

            # Conditionally process missing data correlations if applicable.
            if args.scenario == 'incomplete':
                Corr_miss[0, i] = GFA_output.corrmiss

            # Calculate and update the total variance explained if not already present.
            if not hasattr(GFA_output, 'VarExp_total'):
                total_variance, factors_variance = calculate_variance_explained(GFA_output, args)
                GFA_output.VarExp_total = total_variance
                GFA_output.VarExp_factors = factors_variance

                # Save the updated GFA output object back to the file.
                with open(result_dict_file, 'wb') as parameter_file:
                    pickle.dump(GFA_output, parameter_file)

            # Log the percentage of variance explained by the factors.
            variance_explained_pct = (GFA_output.VarExp_factors / GFA_output.VarExp_total) * 100
            print(f'Percentage of variance explained by the estimated factors: '
                f'{np.around(variance_explained_pct, 2)}', file=ofile)

            # Find and log the most relevant factors.
            relevant_shared_factors, relevant_specific_factors = get_factors(GFA_output, res_path)
            print(f'Relevant shared factors: {np.array(relevant_shared_factors) + 1}', file=ofile)
            for group_index in range(args.num_groups):
                group_factors = np.array(relevant_specific_factors[group_index]) + 1
                print(f'Relevant specific factors (group {group_index + 1}): {group_factors}', file=ofile)
        else:
            # If the results file does not exist or is empty, print an error message.
            print(f'Error: Results file {result_dict_file} is missing or empty.', file=ofile)


    best_ELBO = int(np.argmax(ELBO)+1)
    print('\nBest model', file=ofile)  
    print('-' * 48, file=ofile)  # Create a separator line.
    print('Best ELBO: ', best_ELBO, file=ofile)

    filepath = f'{res_path}[{best_ELBO}]Results.dictionary'
    with open(filepath, 'rb') as parameters:
        GFA_best = pickle.load(parameters)

    # Plot and save the ELBO curve for the best model
    plt.figure()
    plt.title('ELBO Over Iterations for the Best Model')
    plt.plot(GFA_best.L[1:])  # Exclude the first value if it's an outlier or initialization value
    elbo_figure_path = os.path.join(res_path, 'ELBO.png')
    plt.savefig(elbo_figure_path)
    plt.close()

    # Determine the relevant factors for the best model
    shared_factor, specific_factor = get_factors(GFA_best, res_path, BestModel=True)

    # Compile indices of brain and NI factors
    brain_factors_indices = sorted(set(shared_factor + specific_factor[0])) 
    NI_factors_indices = sorted(set(shared_factor + specific_factor[1]))

    # Log the relevant factors
    print(f'Brain factors: {np.array(brain_factors_indices) + 1}', file=ofile)  # Adding 1 for 1-based indexing
    print(f'NI factors: {np.array(NI_factors_indices) + 1}', file=ofile)

    # Save factors to files if present
    if brain_factors_indices:
        brain_factors = {'wx1': GFA_best.means_w[0][:, brain_factors_indices]}
        io.savemat(os.path.join(res_path, 'wx1.mat'), brain_factors)

    if NI_factors_indices:
        NI_factors = {'wx2': GFA_best.means_w[1][:, NI_factors_indices]}
        io.savemat(os.path.join(res_path, 'wx2.mat'), NI_factors)

    # Save relevant latent factors
    latent_factors_indices = sorted(set(brain_factors_indices + NI_factors_indices))
    latent_factors = {'Z': GFA_best.means_z[:, latent_factors_indices]}
    io.savemat(os.path.join(res_path, 'Z.mat'), latent_factors)

    # Log multi-output predictions
    print('\nMulti-output predictions:', file=ofile)
    print('------------------------------------------------', file=ofile)
    sorted_indices_by_mse = np.argsort(np.mean(MSE_NI_test, axis=0))
    top_predictors_count = 10
    print(f'Top {top_predictors_count} predicted variables:', file=ofile)
    for index in range(top_predictors_count):
        print(y_labels[sorted_indices_by_mse[index]], file=ofile)

    # If handling incomplete data, log predictions for missing data
    if args.scenario == 'incomplete':
        print('\nPredictions for missing data:', file=ofile)
        print('------------------------------------------------', file=ofile)
        avg_correlation = np.mean(Corr_miss)
        std_correlation = np.std(Corr_miss)
        print(f'Pearson correlation (avg ± std): {avg_correlation:.3f} (± {std_correlation:.3f})', file=ofile)

    plot_MSE(res_path, MSE_NI_test, MSE_NI_train, y_labels)

    ofile.close()
    print('Visualisation concluded!')    



