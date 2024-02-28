import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
    covar = np.zeros([len(Z[0]), len(Z[0])])

    for i in range(len(covar)):
        for j in range(len(covar)):
            for k in range(len(Z)):
                covar[i][j] += Z[k][i] * Z[k][j]

    return covar

def find_pcs(cov):
    eigenval, eigenvec = np.linalg.eig(cov)
    print(eigenval)
    print(eigenvec)

    idx = eigenval.argsort()[::-1]
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:,idx]

    return eigenvec, eigenval

def project_data(Z, PCS, L):
    # Z* = Zu
    projected = np.zeros(len(Z))

    u = PCS[0]

    for i in range(len(Z)):
        for j in range(len(u)):
            projected[i] += Z[i][j]*u[j]
    
    return projected


def show_plot(Z, Z_star):
    fig, (ax1) = plt.subplots(1)
    
    ax1.scatter(Z[:,0], Z[:,1])
    ax1.scatter(Z_star, np.zeros(len(Z_star)))
    ax1.axhline(y = 0, color = 'r', linestyle = '-')

    plt.show()