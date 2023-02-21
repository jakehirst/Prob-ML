import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

a = [1, 5, 10]  
b = [1, 5, 10]  

for i in range(len(a)):
    x = np.linspace(0, 1, 1000)

    y = beta.pdf(x, a[i], b[i])
    plt.plot(x, y,
        label=f'beta pdf a = {a[i]}, b = {b[i]}')

plt.legend()
#plt.xlim((-10,10))
plt.title("Beta distribution a = b")
plt.xlabel('x')
plt.ylabel('Prob density function')
plt.savefig("hw1/Beta_distribution_equal_ab")
plt.close()

a = [1, 5, 10]  
b = [2, 6, 11]  

for i in range(len(a)):
    x = np.linspace(0, 1, 1000)

    y = beta.pdf(x, a[i], b[i])
    plt.plot(x, y,
        label=f'beta pdf a = {a[i]}, b = {b[i]}')

plt.legend()
#plt.xlim((-10,10))
plt.title("Beta distribution b = a+1")
plt.xlabel('x')
plt.ylabel('Prob density function')
plt.savefig("hw1/Beta_distribution_b_greater_a")
