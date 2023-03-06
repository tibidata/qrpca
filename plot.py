from models import PCA
import numpy as np

"""
Script to load the test file and to plot the necessary plots.
The test file was produced with numpy's built in function np.random.rand() with 500 rows and 3 columns
"""

C = np.loadtxt('test_3dim.csv', delimiter=',')

n_components = [1, 2, 3]

reg_params = [0, 1, 3, 6.2, 6.3, 6.6]

for i in range(len(n_components)):
    for j in range(len(reg_params)):
        pca = PCA(n_components=n_components[i], regularized=True, reg_parameter=reg_params[j])
        pca.transform(C)
        pca.plot()
