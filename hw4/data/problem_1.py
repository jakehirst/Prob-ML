import sys
import os
import numpy as np
import math as m
import matplotlib.pyplot as plt

x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "data/example-code/")
from gmq_example import gass_hermite_quad

#******* part a *********

def inside_sigmoid(x):
    return 10 * x + 3

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

"""scalar distribution for problem 1"""
def my_func(x):
    f_x = np.exp(-x**2) * sigmoid(inside_sigmoid(x))
    return sigmoid(f_x)

def my_func_no_exp(x):
    return sigmoid(inside_sigmoid(x))

degree = 200
N = gass_hermite_quad( my_func_no_exp, degree=degree)
print(f"\nNormalization constant for part 1a = {N}\n")

num_points = 5000
x = np.linspace(-5, 5, num_points)
y = np.exp(-x**2) * sigmoid(inside_sigmoid(x)) / N
plt.plot(x,y, label='part 1a density curve p(z)')
plt.xlim(-5,5)
plt.xlabel("z")
plt.ylabel("p(z)")

#******* part b *********

max = max(y)
max_index = np.argmax(y)
theta_0 = (max_index / num_points) * 10 - 5
# A is the negative second derivative of log of the scalar distribution function
A = -(-2 - (100*np.exp(-10*theta_0 - 3) / (1 + np.exp(-10*theta_0 - 3))**2))
approx_y =  max * np.exp(- 0.5 * np.power(x-theta_0, 2) * A)
variacne = 1/A
print("part b:")
print(f"mean = {theta_0}")
print(f"variance = {variacne}")
plt.plot(x, approx_y, label='part 1b laplace approzimation')

#******* part c *********
def Lambda(xi):
    left = -1/(2*inside_sigmoid(xi))
    right = sigmoid(inside_sigmoid(xi)) - 0.5
    return left * right

def E_step(x, xi):
    left = np.exp(-x**2) * sigmoid(inside_sigmoid(xi))
    lam_x = Lambda(xi)
    right= np.exp(5*(x-xi) + lam_x * (10*(x-xi)) * (10*(x + xi) + 6))
    loc_var = left * right/N
    return loc_var

def M_step(x, loc_var):
    xi_new  = x[loc_var.argmax()]
    xi = xi_new
    return xi

xi = 0
for i in range(100):
    loc_var = E_step(x, xi)
    xi = M_step(x, loc_var)
    
plt.plot(x, loc_var, label= 'local variational inference')


plt.title('problem 1 plots')
plt.legend()
plt.show()
plt.close()







