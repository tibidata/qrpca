from matplotlib import pyplot as plt
import numpy as np


class PCA:
    def __init__(self, n_components: int, regularized: bool = False, reg_parameter: int = 0):

        """

        :param n_components: The rank we want to approximate our matrix with.
        :param regularized: If True we use quadratically regularized PCA if False normal PCA. False by default
        :param reg_parameter: If we use QRPCA this is the regularization parameter.
        """
        self.reduced_matrix = None
        self.centralized_matrix = None
        self.n_components = n_components
        self.regularized = regularized
        self.reg_parameter = reg_parameter

    def transform(self, matrix_to_transform):
        """
        Transforms a matrix into a k-rank approximation.
        :param matrix_to_transform: The original matrix to transform.
        :return: The reduced matrix and the centered matrix.
        """
        if not self.regularized:
            mean_point = matrix_to_transform.mean(axis=0)
            self.centralized_matrix = matrix_to_transform - mean_point

            u, s, vt = np.linalg.svd(self.centralized_matrix, full_matrices=False)

            pcs = u @ np.diag(s)

            self.reduced_matrix = pcs[:, :self.n_components] @ vt[:self.n_components]

            return self.reduced_matrix, self.centralized_matrix

        elif self.regularized:

            mean_point = matrix_to_transform.mean(axis=0)
            self.centralized_matrix = matrix_to_transform - mean_point

            u, s, vt = np.linalg.svd(self.centralized_matrix, full_matrices=False)

            reg_matrix = np.ones_like(s) * self.reg_parameter

            s_reg = s - reg_matrix

            s_reg[s_reg < 0] = 0

            pcs = u @ np.diag(s_reg)

            self.reduced_matrix = pcs[:, :self.n_components] @ vt[:self.n_components]

            return self.reduced_matrix, self.centralized_matrix

    def plot(self):
        """
        Plots the reduced matrix and the centralized matrix
        :return: None
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', )
        ax.scatter(self.reduced_matrix[:, 0], self.reduced_matrix[:, 1], self.reduced_matrix[:, 2], c='r')
        ax.scatter(self.centralized_matrix[:, 0], self.centralized_matrix[:, 1], self.centralized_matrix[:, 2], c='b',
                   alpha=0.2)
        plt.title('number of components = ' + str(self.n_components) + 'reg. param. =' + str(self.reg_parameter))

        plt.show()
