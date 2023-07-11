import numpy as np
from scipy.stats import norm
import math
import function
import matplotlib.pyplot as plt

T = 100
xx0 = np.arange(-2, 2+4/(T-1), 4/(T-1))

xx = np.mat(xx0)
#xx = np.random.uniform(0,1, T)
xx = xx.T
uu = np.mat(norm.cdf(xx, 0, 1))
v = np.arange(0.02, 0.2+0.03, 0.03)

# 1. Normal copula
rho = [0.5]
zz = np.mat(np.ones((T, T))*(-999.99))

for ii in range(0, T, 1):
    zz[:, ii] = function.bivnormpdf(xx, xx[ii, :], [0, 0], function.theta2rho(rho))

# plot the figure
fig1 = plt.figure(1)
plt.contour(xx0, xx0, zz, colors='deepskyblue')
plt.title('Normal copula, rho = 0.5')

# 2. Clayton copula
kappa = 1

zz = np.mat(norm.pdf(xx)*np.ones((1, T)))
for ii in range(0, T, 1):
    zz[:, ii] = np.multiply(np.multiply(zz[:, ii], norm.pdf(xx[ii])), function.claytonpdf(uu, uu[ii], kappa))

# plot the figure
fig3 = plt.figure(3)
plt.contour(xx0, xx0, zz, colors='deepskyblue')
plt.title('Clayton copula, kappa = 1')
fig4 = plt.figure(4)
print('complete')








