import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

mu = 0  # Mean
std = 1  # Standard deviation
deg_of_freedom = [0.1, 1, 10, 100, 10000000]

# Creating x values for student t dsitribution
x = np.linspace(t.ppf(0.0000000000000001, deg_of_freedom[-1], loc=mu, scale=std),
                t.ppf(0.9999999999999999, deg_of_freedom[-1], loc=mu, scale=std), 1000)


#Plotting the t-distribution
for df in deg_of_freedom:
    y = t.pdf(x, df, loc=mu, scale=std)
    plt.plot(x, y, label=f'student t, df={df}')

#Creating x values for Gaussian distribution
x = np.linspace(norm.ppf(0.0000000000000001),
                norm.ppf(0.9999999999999999), 100)

#Plotting Gaussian distribution
plt.plot(x, norm.pdf(x),
       'y--', lw=5, alpha=0.6, label='Gaussian distribution')


plt.legend()
plt.xlim((-10,10))
plt.title("Student's t vs Gaussian distribution")
plt.xlabel('x')
plt.ylabel('Prob density function')
plt.savefig("hw1/Student's_t_vs_Gaussian")
plt.show()

