import numpy as np
import math
def theta2rho(theta):
    m = len(theta)
    k = (1+np.sqrt(1+8*m))/2
    if np.mod(k, 1) != 0:
        print('vector is not of correct length')
    else:
        k = int(k)
    out1 = np.ones((k, k))*(-999.99)
    counter = 0
    for ii in range(0, k, 1):
        for jj in range(ii, k, 1):
            if ii == jj:
                out1[ii][jj] = 1
            else:
                out1[ii][jj] = theta[counter]
                out1[jj][ii] = theta[counter]
                counter = counter+1
    return out1


def bivnormpdf(X, Y, MU, VCV):
    k = 2
    detVCV = np.linalg.det(VCV)
    invVCV = np.linalg.inv(VCV)
    T = max(X.shape[0],1)
    N = 1

    if T >= N:
        if X.shape[0] < T:
            X = X[0][0]*np.ones((T, 1))
        if 1 < T:
            Y = float(Y)*np.ones((T, 1))
            Y = np.mat(Y)
        X = np.append(X, Y, axis=1)
        out1 = np.ones((T, 1))*(-999.99)
        for tt in range(0, T, 1):
            data =(X[tt] - MU)
            out1[tt] = ((2*np.pi)**(-k/2))*(detVCV**(-0.5))*math.exp(-0.5*data*invVCV*data.T)
    else:
        if X.shape[1] < N:
            X = X[0][0]*np.ones(1, N)
            Y = Y*np.ones(1, N)
        X = np.mat(np.append(X, Y, axis=0))
        X = X.T
        out1 = (-999.99)*np.ones((1, N))
        for tt in range(0, N, 1):
            data = np.mat(X[tt] - MU)
            out1[tt] = (2 * np.pi) ** (-k / 2) * detVCV ** (-0.5) * math.exp(-0.5 * (X[tt] - MU) * invVCV * data.T)
    return out1

#def tcopulapdf(U, V, RHO, NU):

def claytonpdf(u, v,k1):
    T = max(u.shape[0], v.shape[0], 1)
    if u.shape[0] < T:
        u = np.mat(u*np.ones((T, 1)))
    if v.shape[0] < T:
        v = np.mat(float(v)*np.ones((T, 1)))
    if 1 < T:
        k1 = np.mat(k1*np.ones((T, 1)))
    uv = np.multiply(u, v)
    a = (np.zeros((T, 1)))
    b = (np.zeros((T, 1)))
    c = (np.zeros((T, 1)))
    d = (np.zeros((T, 1)))
    for i in range(0, T, 1):
        a[i] = np.array(uv[i])**np.array(-k1[i]-1)
        b[i] = np.array(u[i])**np.array(-k1[i])
        c[i] = np.array(v[i])**np.array(-k1[i])
    for i in range(0, T, 1):
        d[i] = np.array(b[i]+c[i]-1)**np.array(-2-1/k1[i])
    pdf = np.multiply(np.multiply(1+k1, a), d)
    return pdf









