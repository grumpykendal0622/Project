""" GFA predictor. Predict output for specified groups & missing values in the dataset """

import numpy as np

class GFAtools(object):
    def __init__(self, X, model):
        self.X = X
        self.model = model
    
    def _update_sigmaZ(self, sigmaZ, noise, index):
        """
        Update covariance matrix of the latent variable Z for predicting groups.

        Parameters:
        - sigmaZ: The current value of the covariance matrix of Z.
        - noise: A string indicating the type of noise model in use.
        - index: An integer indicating the current group or view being processed.

        Returns:
        - The updated covariance matrix sigmaZ.
        """
        if 'spherical' in noise:
            sigmaZ += self.model.E_tau[0, index] * self.model.E_WW[index]
        else:
            for j in range(self.model.d[index]):
                w = self.model.means_w[index][j, :].reshape((1, self.model.k))
                sigmaZ += self.model.E_tau[index][0, j] * (
                    self.model.sigma_w[index][:, :, j] + w.T @ w)
        return sigmaZ

    def _update_meanZ(self, meanZ, noise, index, N):
        """
        Update the mean of the latent variable Z  for predicting groups.

        Parameters:
        - meanZ: The current value of the mean matrix of Z.
        - noise: A string indicating the type of noise model in use.
        - index: An integer indicating the current group being processed.
        - N: The number of samples.

        Returns:
        - The updated mean matrix meanZ.
        """        
        if 'spherical' in noise:
            meanZ += self.X[index] @ self.model.means_w[index] * self.model.E_tau[0, index]
        else:
            for j in range(self.model.d[index]):
                w = self.model.means_w[index][j, :].reshape((1, self.model.k))
                x = self.X[index][:, j].reshape((N, 1))
                meanZ += x @ w * self.model.E_tau[index][0, j]
        return meanZ

    def _calculate_sigmaZ(self, t_tmp, N):
        """
        Calculate the covariance matrix sigmaZ for predicting missing values.

        Parameters:
        - t_tmp: A list of indices representing specific groups or views in the data.
        - N: The number of data points or samples.

        Returns:
        - sigmaZ: A list of N covariance matrices for the latent variable Z.
        """
        sigmaZ = np.array([np.identity(self.model.k) for _ in range(N)])
        for t in t_tmp:
            for j in range(self.model.d[t]):
                not_nan_indices = ~np.isnan(self.X[t][:, j])
                w = self.model.means_w[t][j, :].reshape((1, self.model.k))
                ww = self.model.sigma_w[t][:, :, j] + w.T @ w
                sigmaZ[not_nan_indices] += self.model.E_tau[t][0, j] * ww
        return sigmaZ

    def _calculate_meanZ(self, sigmaZ, t_tmp, N):   
        """
        Calculate the mean vector meanZ for the latent variable Z for predicting missing values.

        Parameters:
        - sigmaZ: A list of N covariance matrices for the latent variable Z.
        - t_tmp: A list of indices representing specific groups or views in the data.
        - N: The number of data points or samples.

        Returns:
        - meanZ: The mean matrix for the latent variable Z.
        """
        meanZ = np.zeros((N, self.model.k))
        for t in t_tmp:
            for j in range(self.model.d[t]):
                not_nan_indices = ~np.isnan(self.X[t][:, j])
                w = self.model.means_w[t][j, :].reshape((1, self.model.k))
                x = self.X[t][not_nan_indices, j]
                S = x[:, None] * w * self.model.E_tau[t][0, j]
                # Loop over each sample to invert the corresponding sigmaZ matrix
                for i in np.where(not_nan_indices)[0]:
                    meanZ[i, :] += S[i, :] @ np.linalg.inv(sigmaZ[i])
        return meanZ

    def PredictGroups(self, obs_ds, noise):
        """
        Predict output for specified groups in the dataset.

        Parameters:
        - obs_ds: An array indicating observed (1) and non-observed/predicted (0) datasets.
        - noise: The noise model used in the prediction.

        Returns:
        - A list of arrays with predictions for each group in the 'pred' list.
        """
        train = np.where(obs_ds == 1)[0]
        pred = np.where(obs_ds == 0)[0]
        N = self.X[0].shape[0]

        sigmaZ = np.identity(self.model.k)
        for i in train:
            sigmaZ = self._update_sigmaZ(sigmaZ, noise, i)

        w, v = np.linalg.eig(sigmaZ)
        sigmaZ_inv = v @ np.diag(1/w) @ v.T

        meanZ = np.zeros((N, self.model.k))
        for i in train:
            meanZ = self._update_meanZ(meanZ, noise, i, N)

        meanZ = meanZ @ sigmaZ_inv

        return [meanZ @ self.model.means_w[i].T for i in pred]

    def PredictMissing(self, infoMiss):
        """
        Predict missing values in the dataset.

        Parameters:
        - infoMiss: A dictionary containing indices of datasets for which the missing values need to be predicted.

        Returns:
        - X_pred: A list of arrays with predicted values in place of missing values.
        """
        pred = infoMiss['ds']
        train = np.arange(self.model.s)
        N = self.X[0].shape[0]
        X_pred = [np.full((N, self.model.d[p_ind - 1]), np.nan) for p_ind in pred]


        for p, p_ind in enumerate(pred):
            t_tmp = np.delete(train, p_ind - 1)

            sigmaZ = self._calculate_sigmaZ(t_tmp, N)
            meanZ = self._calculate_meanZ(sigmaZ, t_tmp, N)

            # Predict missing values only
            missing_indices = np.isnan(self.X[p_ind -1])
            X_pred[p][missing_indices] = (meanZ @ self.model.means_w[p_ind-1].T)[missing_indices]

        return X_pred
