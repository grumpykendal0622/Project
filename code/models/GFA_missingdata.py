""" Implement proposed GFA model."""

import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import fmin_l_bfgs_b as lbfgsb


class GFA_DiagonalNoiseModel(object):

    def __init__(self, X, params, imputation=False):
        
        self.N = X[0].shape[0] # number of samples
        self.k = params['K']   # number of factors
        self.s = params['num_groups'] # number of groups
        self.d = [X[i].shape[1] for i in range(self.s)] # number of variables in each group
        self.td = np.sum(self.d) # total number of features
        self.scenario = 'complete' if imputation else params['scenario'] # handle imputation
        self.DoRotation = True # handle rotation optimization
        # Initial value of hyperparameters
        self.Alpha_a0 = self.Alpha_b0 = self.Tau_a0 = self.Tau_b0 = 1e-14

        # Handle incomplete dataset and initialize variational parameters
        self._process_missing_data(X)
        self._initialize_Z()
        self._initialize_W()
        self._initialize_Alpha()
        self._initialize_Tau()
        self._initialize_ELBO_constants()


    def _process_missing_data(self, X):
        """
        Process the missing data for each modality.

        If the scenario is 'incomplete', identify the missing entries in each modality 
        and store a binary matrix indicating missing data locations. Also, calculate 
        the count of non-missing entries for each modality.
        """
        if self.scenario == 'incomplete':
            self.X_incomplete = [np.where(np.isnan(X[i]), 1, 0) for i in range(self.s)]
            self.N_incomplete = [np.sum(~np.isnan(X[i]), axis=0) for i in range(self.s)]


    def _initialize_Z(self):
        """
        Initialize the latent variable Z.

        Z is initialized to be multivariate normal with mean zero and identity covariance.
        This function initializes Z with random normal values and sets up the related
        covariance structures.
        """
        self.means_z = np.random.normal(0, 1, size=(self.N, self.k))
        self.sigma_z = np.zeros((self.k, self.k, self.N))
        self.sum_sigmaZ = self.N * np.identity(self.k)

    def _initialize_W(self):
        """
        Initialize the weight matrix W for each modality.

        W is initialized to zero means and zero covariance matrices, setting up 
        the necessary data structures for further computations.
        """
        self.means_w= [np.zeros((self.d[i], self.k)) for i in range(self.s)]
        self.sigma_w = [np.zeros((self.k, self.k, self.d[i])) for i in range(self.s)]
        self.E_WW = [[] for _ in range(self.s)]
        self.Lqw = [[] for _ in range(self.s)]
        
    def _initialize_Alpha(self):
        """
        Initialize the ARD parameter Alpha for each modality.

        Alpha is defined by a shape and scale, and its expectation is calculated.
        The initialization follows the given prior hyperparameters.
        """
        self.a_alpha = [self.Alpha_a0 + self.d[i]/2.0 for i in range(self.s)]
        self.b_alpha = [np.ones((1, self.k)) for _ in range(self.s)]
        self.E_alpha = [self.a_alpha[i] / self.b_alpha[i] for i in range(self.s)]
        
    def _initialize_Tau(self):
        """
        Initialize the precision parameters Tau for each modality.

        Tau accounts for the noise in the observations, and is set to a high 
        expectation indicating low initial noise assumption. The scale parameter
        is initialized to zero, while the shape parameter depends on the 
        scenario being complete or incomplete.
        """
        self.a_tau = [self.Tau_a0 + (self.N_incomplete[i] if self.scenario == 'incomplete' else self.N * np.ones((1,self.d[i]))) / 2 for i in range(self.s)]
        self.b_tau = [np.zeros((1, self.d[i])) for i in range(self.s)]
        self.E_tau = [1000.0 * np.ones((1, self.d[i])) for i in range(self.s)]
        
    def _initialize_ELBO_constants(self):
        """
        Initialize the constants for the Evidence Lower Bound (ELBO) computation.

        These constants include log-alpha and log-tau for each modality, and the 
        constant term in the ELBO which is dependent on the number of observed 
        values and the dimensionality of each modality.
        """
        self.logalpha = [[] for _ in range(self.s)]
        self.logtau = [[] for _ in range(self.s)]
        self.L_const = [-0.5 * (np.sum(self.N_incomplete[i]) if self.scenario == 'incomplete' else self.N * self.d[i]) * np.log(2*np.pi) for i in range(self.s)]
  
    def _update_w(self, X):
        """ 
        Update the variational parameters of the loading matrices.

        Parameters
        ----------
        X : list 
            List of arrays containing the data sample of each group.          
        """
        self.sum_sigmaW = [np.zeros((self.k, self.k)) for _ in range(self.s)]
        
        for i in range(self.s):
            self.Lqw[i] = np.zeros((1, self.d[i]))
            for j in range(self.d[i]):
                if self.scenario != 'complete':
                    indices = np.where(self.X_incomplete[i][:, j] == 0)[0]  
                    # Use the indices to extract the relevant slices and reshape if necessary
                    x = X[i][indices, j].reshape(1, -1)  # Using -1 in reshape infers the size from the length of the array
                    Z = self.means_z[indices, :].reshape(len(indices), self.k)
                else:
                    x = np.reshape(X[i][:, j], (1, X[i].shape[0]))
                    Z = self.means_z
                
                # Common computations for both scenarios
                S1 = self.sum_sigmaZ + np.dot(Z.T, Z)
                S2 = np.dot(x, Z)
                
                # Update covariance matrices of Ws    
                self.sigma_w[i][:, :, j] = np.diag(self.E_alpha[i]) + self.E_tau[i][0, j] * S1
                cho = np.linalg.cholesky(self.sigma_w[i][:, :, j])
                invCho = np.linalg.inv(cho)
                self.sigma_w[i][:, :, j] = np.dot(invCho.T, invCho)
                self.sum_sigmaW[i] += self.sigma_w[i][:, :, j]
                
                # Update expectations of Ws
                self.means_w[i][j, :] = np.dot(S2, self.sigma_w[i][:, :, j]) * self.E_tau[i][0, j]
                
                # Compute determinant for ELBO
                self.Lqw[i][0, j] = -2 * np.sum(np.log(np.diag(cho)))
            
            # Calculate E[W^T W]
            self.E_WW[i] = self.sum_sigmaW[i] + np.dot(self.means_w[i].T, self.means_w[i])

    def _update_z(self, X):
        """
        Update the variational parameters of the latent variables.

        Parameters
        ----------
        X : list of np.ndarray
            List of arrays containing the data sample of each group.
        """

        self.means_z = np.zeros_like(self.means_z)
        if self.scenario == 'complete':
            self._update_z_complete_scenario(X)
        else:
            self._update_z_incomplete_scenario(X)
        # Calculate E[Z^T Z]
        self.E_zz = self.sum_sigmaZ + self.means_z.T @ self.means_z

    def _update_z_complete_scenario(self, X):
        """Helper function to update parameters in the 'complete' scenario."""
        self.sigma_z = np.identity(self.k)
        self.means_z = np.zeros_like(self.means_z)

        for i in range(self.s):
            for j in range(self.d[i]):
                weight_mean = self.means_w[i][j, :].reshape((1, self.k))
                weight_second_moment = self.sigma_w[i][:, :, j] + np.dot(weight_mean.T, weight_mean)
                self.sigma_z += weight_second_moment * self.E_tau[i][0, j]

        # Efficient way of computing sigmaZ
        chol_sigma_z = np.linalg.cholesky(self.sigma_z)
        inv_chol_sigma_z = np.linalg.inv(chol_sigma_z)
        self.sigma_z = inv_chol_sigma_z.T @ inv_chol_sigma_z
        self.sum_sigmaZ = self.N * self.sigma_z

        # Update expectations of Z
        for i in range(self.s):
            for j in range(self.d[i]):
                x = X[i][:, j].reshape((self.N, 1))
                weight_mean = self.means_w[i][j, :].reshape((1, self.k))
                self.means_z += x @ weight_mean * self.E_tau[i][0, j]
        
        self.means_z = self.means_z @ self.sigma_z
        # Compute determinant for ELBO
        self.Lqz = -2 * np.sum(np.log(np.diag(chol_sigma_z)))

    def _update_z_incomplete_scenario(self, X):
        """Helper function to update parameters in the incomplete scenario."""
        self.sigma_z = np.zeros((self.k, self.k, self.N))
        self.sum_sigmaZ = np.zeros((self.k, self.k))
        self.Lqz = np.zeros((1, self.N))
        
        for n in range(self.N):
            self.sigma_z[:, :, n] = np.identity(self.k)
            weighted_sum = np.zeros((1, self.k))
            for i in range(self.s):
                observed_dims = np.where(self.X_incomplete[i][n, :] == 0)[0]
                for j in observed_dims:
                    weight_mean = self.means_w[i][j, :].reshape((1, self.k))
                    weight_second_moment = self.sigma_w[i][:, :, j] + weight_mean.T @ weight_mean
                    self.sigma_z[:, :, n] += weight_second_moment * self.E_tau[i][0, j]
                x = X[i][n, observed_dims].reshape((1, len(observed_dims)))
                tau = self.E_tau[i][0, observed_dims].reshape((1, len(observed_dims)))
                weighted_sum += x @ np.diag(tau[0]) @ self.means_w[i][observed_dims, :]
            
            # Update covariance matrix of Z
            chol_sigma_z = np.linalg.cholesky(self.sigma_z[:, :, n])
            inv_chol_sigma_z = np.linalg.inv(chol_sigma_z)
            self.sigma_z[:, :, n] = inv_chol_sigma_z.T @ inv_chol_sigma_z
            self.sum_sigmaZ += self.sigma_z[:, :, n]
            
            # Update expectations of Z
            self.means_z[n, :] = weighted_sum @ self.sigma_z[:, :, n]
            
            # Compute determinant for ELBO
            self.Lqz[0, n] = -2 * np.sum(np.log(np.diag(chol_sigma_z)))

    def _update_alpha(self):
        """
        Recalculate the variational parameters for the alpha distribution.

        The method iterates through each group, updating the b_alpha parameter based on the expected value of the W matrix.
        It then computes the expected value of the alpha parameter.

        Parameters
        ----------
        None
            
        """
        self.b_alpha = [self.Alpha_b0 + 0.5 * np.diag(e_ww) for e_ww in self.E_WW]
        self.E_alpha = [a / b for a, b in zip(self.a_alpha, self.b_alpha)]

    def _update_tau(self, X):
        """
        Update the variational parameters of the precision parameters, tau.
        
        This function optimizes the variational parameters associated with the precision of the likelihood for each
        feature in each group of data. It accounts for different scenarios of data completeness and ensures the
        updates are accurate whether data is missing or fully observed.

        Parameters
        ----------
        X : list of np.ndarray
            List of arrays, each corresponding to a group of data. Each array is a matrix with dimensions N x d_i,
            where N is the number of data samples, and d_i is the number of features in the i-th group.
        """
        for i in range(self.s):
            for j in range(self.d[i]):
                mean_weight = self.means_w[i][j, :].reshape((1, self.k))
                cov_weight = self.sigma_w[i][:, :, j] + np.dot(mean_weight.T, mean_weight)
                
                if self.scenario == 'complete':
                    data_feature = X[i][:, j].reshape((self.N, 1))
                    mean_z = self.means_z
                    cov_z = self.E_zz
                else:
                    row_indices = np.where(self.X_incomplete[i][:, j] == 0)[0]
                    data_feature = X[i][row_indices, j].reshape((len(row_indices), 1))
                    mean_z = self.means_z[row_indices, :]
                    sum_cov_z = np.sum(self.sigma_z[:, :, row_indices], axis=2)
                    cov_z = sum_cov_z + np.dot(mean_z.T, mean_z)

                # Update b_tau        
                self.b_tau[i][0, j] = self.Tau_b0 + 0.5 * (
                    np.dot(data_feature.T, data_feature) +
                    np.trace(np.dot(cov_weight, cov_z)) -
                    2 * np.dot(data_feature.T, mean_z).dot(mean_weight.T)
                )

            # Update expectation of tau             
            self.E_tau[i] = self.a_tau[i] / self.b_tau[i]


    def _calculate_ELBO(self, X):
        
        """ 
        Calculate Evidence Lower Bound (ELBO).

        Parameters
        ----------
        X : list 
            List of arrays containing the data sample of each group.

        Returns
        -------
        L : float
            Evidence Lower Bound (ELBO).              
        
        """     
        # Calculate E[ln p(X|Z,W,tau)]
        L = 0
        # for i in range(0, self.s):
        #     #calculate E[ln alpha] and E[ln tau]
        #     self.logalpha[i] = digamma(self.a_alpha[i]) - np.log(self.b_alpha[i])
        #     self.logtau[i] = digamma(self.a_tau[i]) - np.log(self.b_tau[i])
        #     if self.scenario == 'complete':
        #         L += self.L_const[i] + np.sum(self.N * self.logtau[i]) / 2 - \
        #         np.sum(self.E_tau[i] * (self.b_tau[i] - self.Tau_b0)) 
        #     else:    
        #         L += self.L_const[i] + np.sum(self.N_incomplete[i] * self.logtau[i]) / 2 - \
        #             np.sum(self.E_tau[i] * (self.b_tau[i] - self.Tau_b0))   
        for i in range(self.s):
            self.logalpha[i] = digamma(self.a_alpha[i]) - np.log(self.b_alpha[i])
            self.logtau[i] = digamma(self.a_tau[i]) - np.log(self.b_tau[i])
            
            N_factor = self.N if self.scenario == 'complete' else self.N_incomplete[i]
            tau_diff = self.E_tau[i] * (self.b_tau[i] - self.Tau_b0)
            
            L += self.L_const[i] + np.sum(self.logtau[i] * N_factor) / 2 - np.sum(tau_diff)


        # # Calculate E[ln p(Z)] - E[ln q(Z)]
        # self.Lpz = - 1/2 * np.sum(np.diag(self.E_zz))
        # if self.scenario == 'complete':
        #     self.Lqz = - self.N * 0.5 * (self.Lqz + self.k)
        # else: 
        #     self.Lqz = - 0.5 * (np.sum(self.Lqz) + self.k)   
        # L += self.Lpz - self.Lqz
        
        # Calculate E[ln p(Z)] and E[ln q(Z)]
        self.Lpz = -0.5 * np.sum(np.diag(self.E_zz))
        self.Lqz = -0.5 * (self.N if self.scenario == 'complete' else np.sum(self.Lqz)) - 0.5 * self.k

        # Update the lower bound with the difference
        L += self.Lpz - self.Lqz


        # # Calculate E[ln p(W|alpha)] - E[ln q(W)]
        # self.Lpw = 0
        # for i in range(0, self.s):
        #     self.Lpw += 0.5 * self.d[i] * np.sum(self.logalpha[i]) - np.sum(
        #         np.diag(self.E_WW[i]) * self.E_alpha[i])
        #     self.Lqw[i] = - 0.5 * np.sum(self.Lqw[i]) - 0.5 * self.d[i] * self.k 
        # L += self.Lpw - sum(self.Lqw)      
        
        # Calculate E[ln p(W|alpha)] - E[ln q(W)]
        self.Lpw = 0
        self.Lpw = sum(0.5 * self.d[i] * np.sum(self.logalpha[i]) -
                    np.sum(np.diag(self.E_WW[i]) * self.E_alpha[i]) for i in range(self.s))

        self.Lqw = [-0.5 * (np.sum(qw) + self.d[i] * self.k) for i, qw in enumerate(self.Lqw)]

        # Update the lower bound with the difference
        L += self.Lpw - sum(self.Lqw)
                     

        # # Calculate E[ln p(alpha) - ln q(alpha)]
        # self.Lpa = self.Lqa = 0
        # for i in range(0, self.s):
        #     self.Lpa += self.k * (-gammaln(self.Alpha_a0) + self.Alpha_a0 * np.log(self.Alpha_b0)) \
        #         + (self.Alpha_a0 - 1) * np.sum(self.logalpha[i]) - self.Alpha_b0 * np.sum(self.E_alpha[i])
        #     self.Lqa += -self.k * gammaln(self.a_alpha[i]) + self.a_alpha[i] * np.sum(np.log(
        #         self.b_alpha[i])) + ((self.a_alpha[i] - 1) * np.sum(self.logalpha[i])) - \
        #         np.sum(self.b_alpha[i] * self.E_alpha[i])         
        # L += self.Lpa - self.Lqa               
        # Calculate E[ln p(alpha) - ln q(alpha)]
        self.Lpa = self.Lqa = 0
        k_a0_term = self.k * (-gammaln(self.Alpha_a0) + self.Alpha_a0 * np.log(self.Alpha_b0))
        a0_minus_1 = self.Alpha_a0 - 1
        sum_logalpha = np.sum(self.logalpha, axis=1)  # Sum over each group of logalpha once, since it's used multiple times.
        sum_E_alpha = np.sum(self.E_alpha, axis=1)    # Same for E_alpha

        for i in range(self.s):
            self.Lpa += k_a0_term + a0_minus_1 * sum_logalpha[i] - self.Alpha_b0 * sum_E_alpha[i]
            self.Lqa += -self.k * gammaln(self.a_alpha[i]) + \
                        self.a_alpha[i] * np.sum(np.log(self.b_alpha[i])) + \
                        (self.a_alpha[i] - 1) * sum_logalpha[i] - \
                        np.sum(self.b_alpha[i] * self.E_alpha[i])

        L += self.Lpa - self.Lqa

        # Calculate E[ln p(tau) - ln q(tau)]
        self.Lpt = self.Lqt = 0
        for i in range(0, self.s):
            self.Lpt +=  self.d[i] * (-gammaln(self.Tau_a0) + self.Tau_a0 * np.log(self.Tau_b0)) \
                + (self.Tau_a0 -1) * np.sum(self.logtau[i]) - self.Tau_b0 * np.sum(self.E_tau[i])
            self.Lqt += -np.sum(gammaln(self.a_tau[i])) + np.sum(self.a_tau[i] * np.log(self.b_tau[i])) + \
                np.sum((self.a_tau[i] - 1) * self.logtau[i]) - np.sum(self.b_tau[i] * self.E_tau[i])         
        L += self.Lpt - self.Lqt
                

        return L
   
           
    def _optimize_rotation(self):
        """
        Optimize the rotation to enhance the alignment of latent variables across different groups. 
        This is achieved by flattening the identity matrix of size kxk and finding the optimal rotation 
        using the L-BFGS-B algorithm. Once the optimal rotation is achieved, transformation matrices 
        are updated, and dependent variables are adjusted accordingly. If optimization fails, the rotation 
        process is stopped.
        """
        x0 = np.ravel(np.identity(self.k))
        optimized_result = lbfgsb(self._obj, x0, self._obj_gradient)
        if optimized_result[2]['warnflag'] == 0:
                # Update transformation matrix R
            Rot = np.reshape(optimized_result[0],(self.k,self.k))
            u, s, v = np.linalg.svd(Rot) 
            Rotinv = np.dot(v.T * np.outer(np.ones((1,self.k)), 1/s), u.T)
            det = np.sum(np.log(s)) 
            
            # # Update Z 
            # self.means_z = np.dot(self.means_z, Rotinv.T)
            # if self.scenario == 'complete':
            #     self.sigma_z = np.dot(Rotinv, self.sigma_z).dot(Rotinv.T) 
            # else:    
            #     for n in range(0, self.N):
            #         self.sigma_z[:,:,n] = np.dot(Rotinv, self.sigma_z[:,:,n]).dot(Rotinv.T)
            #         self.sum_sigmaZ += self.sigma_z[:,:,n]
            # self.E_zz = self.sum_sigmaZ + np.dot(self.means_z.T, self.means_z) 
            # self.Lqz += -2 * det  
            self._rotate_Z(Rotinv, det)
            
            # Update W
            self.sum_sigmaW = [np.zeros((self.k,self.k)) for _ in range(self.s)]
            for i in range(0, self.s):
                self.means_w[i] = np.dot(self.means_w[i], Rot)
                for j in range(0, self.d[i]):
                    self.sigma_w[i][:,:,j] = np.dot(Rot.T, 
                        self.sigma_w[i][:,:,j]).dot(Rot)
                    self.sum_sigmaW[i] += self.sigma_w[i][:,:,j]     
                self.E_WW[i] = self.sum_sigmaW[i] + \
                    np.dot(self.means_w[i].T, self.means_w[i])
                self.Lqw[i] += 2 * det
            # self._rotate_W(Rotinv, det)
        else:
            self.DoRotation = False    
            print('Rotation optimization did not converge. Stopping rotation updates.')     

        
        # if optimized_result[2]['warnflag'] == 0:
        #     optimal_rotation_matrix = np.reshape(optimized_result[0], (self.k, self.k))
            
        #     # Singular Value Decomposition
        #     u, singular_values, v_transpose = np.linalg.svd(optimal_rotation_matrix)
            
        #     # Inverse Transformation Matrix
        #     inverse_rotation_matrix = v_transpose.T @ np.diag(1 / singular_values) @ u.T
    
        #     log_determinant = np.sum(np.log(singular_values))
            
        #     # Updating Expectations and Covariances
            # self._rotate_Z(inverse_rotation_matrix, log_determinant)
        #     self._rotate_W(inverse_rotation_matrix, log_determinant)
            
        # else:
        #     self.DoRotation = False
        #     print('Rotation optimization did not converge. Stopping rotation updates.')
            
            
    def _rotate_Z(self, inverse_rotation_matrix, log_determinant):
        """
        Update the latent variables, weights, and associated parameters after a successful rotation optimization.
        
        Parameters
        ----------
        inverse_rotation_matrix : np.ndarray
            The matrix used to transform the latent variables and weights.
        log_determinant : float
            The log determinant of the singular values from the SVD of the optimal rotation matrix.
        """
        self.means_z = self.means_z @ inverse_rotation_matrix.T
        
        if self.scenario == 'complete':
            self.sigma_z = inverse_rotation_matrix @ self.sigma_z @ inverse_rotation_matrix.T
        else:
            for n in range(self.N):
                self.sigma_z[:, :, n] = inverse_rotation_matrix @ self.sigma_z[:, :, n] @ inverse_rotation_matrix.T
                self.sum_sigmaZ += self.sigma_z[:, :, n]
                
        self.E_zz = self.sum_sigmaZ + self.means_z.T @ self.means_z
        self.Lqz += -2 * log_determinant


    def _obj(self, x0):
        """
        Evaluate the negative objective function for the rotation matrix optimization.

        Parameters
        ----------
        x0 : array-like
            Initial value of a flattened rotation matrix R.

        Returns
        -------
        float
            The negative value of the objective function for the current rotation matrix.
        """
        R = x0.reshape(self.k, self.k)
        U, S, _ = np.linalg.svd(R)
        R_inv = U / S
        
        neg_obj_val = -0.5 * np.sum(self.E_zz * (R_inv @ R_inv.T))
        neg_obj_val += (self.td - self.N) * np.sum(np.log(S))
        
        for i in range(self.s):
            group_contrib = R * (self.E_WW[i] @ R)
            neg_obj_val -= self.d[i] * np.sum(np.log(np.sum(group_contrib, axis=0))) / 2

        return -neg_obj_val


    def _obj_gradient(self, r):
        """
        Evaluate the gradient of objective function for the rotation matrix optimization.

        Parameters
        ----------
        x0 : array-like
            Initial value of a flattened rotation matrix R.

        Returns
        -------
        float
            The negative gradient value of the objective function for the current rotation matrix.
        """
        R = r.reshape(self.k, self.k)
        U, S, Vt = np.linalg.svd(R)
        R_inv = np.dot(Vt.T * 1/S, U.T)

        adjustment_matrix = U * (1/S**2)
        gradient_matrix = adjustment_matrix @ U.T @ self.E_zz + np.diag((self.td - self.N) * np.ones(self.k))
        gradient = (gradient_matrix @ R_inv.T).flatten()
        
        for i in range(self.s):
            weighted_R = self.E_WW[i] @ R
            gradient -= self.d[i] * (weighted_R / np.sum(R * weighted_R, axis=0)).flatten()

        return -gradient


    def _trim_factors(self):
        '''
        Trim irrelevant latent factors.
        '''
        # Compute the condition to remove the factors
        col_means_z_sq = np.mean(self.means_z ** 2, axis=0)
        cols_to_keep = col_means_z_sq >= 1e-6
        
        # Filter out the columns across all relevant attributes
        if not np.all(cols_to_keep):
            self.means_z = self.means_z[:, cols_to_keep]
            self.sum_sigmaZ = self.sum_sigmaZ[:, cols_to_keep][cols_to_keep, :]
            self.k = self.means_z.shape[1]

            for i in range(self.s):
                self.means_w[i] = self.means_w[i][:, cols_to_keep]
                self.sigma_w[i] = self.sigma_w[i][:, cols_to_keep, :][cols_to_keep, :, :]
                self.E_WW[i] = self.E_WW[i][:, cols_to_keep][cols_to_keep, :]
                self.E_alpha[i] = self.E_alpha[i][cols_to_keep]
 
    def fit(self, X, max_itr=10000, threshold=1e-6):
        """ 
        Perform Variational Expectation-Maximization(EM) algorithm for maximizing ELBO to infer the true posterior.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.

        max_iter : int
            Maximum number of iterations.

        threshold : float
            Threshold to check model convergence. The model stops when 
            a relative difference in the lower bound falls below this 
            value.                     
        
        """
        ELBO_previous = 0
        self.L = []
        for i in range(max_itr):         
            self._trim_factors()
            self._update_w(X)
            self._update_z(X)
            if i > 0 and self.DoRotation == True:
                self._optimize_rotation()  
            self._update_alpha()
            self._update_tau(X)                
            ELBO_current = self._calculate_ELBO(X)
            self.L.append(ELBO_current)
            diff = ELBO_current - ELBO_previous
            ELBO_previous = ELBO_current
            if abs(diff)/abs(ELBO_current) < threshold:
                print("ELBO (last value):", ELBO_current)
                print("Number of iterations:", i+1)
                self.iter = i+1
                break
            elif i == max_itr:
                print("ELBO did not converge")
            if i < 1:
                print("ELBO (1st value):", ELBO_current)
                

