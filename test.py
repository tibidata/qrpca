from models import PCA
import numpy as np

import matplotlib.pyplot as plt

C = np.loadtxt('test_3dim.csv', delimiter=',')

regulators = [1, 2, 3, 4, 5, 10, 20, 50,100]

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
ax = fig.add_subplot(3,3,projection='3d')
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Quadratically regularized PCA", fontsize=18, y=0.95)

"""for reg_parameter in regulators:

    pca = PCA(n_components=2, regularized=True, reg_parameter=reg_parameter)
    reduced_matrix, centralized_matrix = pca.transform(C)
    ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c='r')
    ax.scatter(centralized_matrix[:, 0], centralized_matrix[:, 1], centralized_matrix[:, 2], c='b', alpha=0.2)
"""
plt.show()
