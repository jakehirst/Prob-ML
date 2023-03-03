import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


""" generates num_examples random x values in a normal distribution between -1 and 1 """
def generate_random_x_values(num_examples):
    np.random.seed(15)
    examples = np.random.uniform(-1.0, 1.0, num_examples)
    return examples

""" ground truth function """
def y(x_arr, w_0, w_1):
    return w_0 + w_1*x_arr

""" adds noise to the y values of our dataset """
def add_noise(y_array, mean, std):
    noise = np.random.normal(mean, std, size=y_array.shape)
    noisy_y_array = y_array + noise
    return noisy_y_array

""" 
    mean of the posterior given to us by 
    m_n = Beta * S_n * x^T * y 
    where S_n is the standard deviation of the posterior
"""
def get_Mn(beta, Sn, x, y):
    Mn = beta * np.matmul(y, np.matmul(np.transpose(x), Sn))
    return Mn

"""
    standard deviation of the posterior given to us by
    S_n ^-1 = alpha * I + beta * x^T * x
"""
def get_Sn(alpha, beta, x, I):
    Sn = np.linalg.inv( alpha * I + beta * np.matmul(x, np.transpose(x)) )
    return Sn

""" gets the posterior by just calling function to get the standard deviation and mean using x, y data """
def get_posterior_mean_and_std(alpha, beta, x, y):
    x = np.vstack((np.ones(len(x)), x))
    Sn = get_Sn(alpha, beta, x, I)
    Mn = get_Mn(beta, Sn, x, y)
    return Mn, Sn

def plot_heatmap(posterior_distribution, title):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, posterior_distribution.pdf(pos))
    ax.set_xlabel("w0")
    ax.set_ylabel("w1")
    ax.set_title(title)
    plt.show()
    plt.close()
    
def plot_distribution_samples(Mn, Sn, num_examples, num_samples, posterior_distribution, temp_x, temp_y):
    samples = multivariate_normal.rvs(mean=Mn, cov=Sn, size=5, random_state=0)
    w0 = samples[:,0] ; w1 = samples[:,1]
    x_line = np.linspace(-1, 1, 20)
    y_line = np.empty((1,0))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(temp_x, temp_y, "r*", label="true")
    for sample in range(len(w0)):
        temp_w0 = w0[sample]
        temp_w1 = w1[sample]    
        y_line = y(x_line, temp_w0, temp_w1)
        ax.plot(x_line, y_line, "b-*", label="true")
    plt.show()
    plt.close()
    return

x_arr = generate_random_x_values(20)
actual_w = [-0.3, 0.5]
y_arr = y(x_arr, actual_w[0], actual_w[1])
noisy_y_arr = add_noise(y_arr, 0, 0.2)

data = pd.DataFrame(dict(x = x_arr, y = noisy_y_arr))

I = np.array([[1, 0], 
              [0, 1]])

prior_mean = np.array([0,0])
prior_std = I*2



#plotting the heatmap of the prior
prior_distribution = multivariate_normal(prior_mean, prior_std)
plot_heatmap(prior_distribution, "heatmap of prior")

alpha = 2
beta = 25
num_samples = 20

num_examples_arr = [1,2,5,20]
for num_examples in num_examples_arr:
    temp_x = x_arr[0:num_examples]
    temp_y = noisy_y_arr[0:num_examples]
    Mn, Sn  = get_posterior_mean_and_std(alpha, beta, temp_x, temp_y)
    posterior_distribution = multivariate_normal(Mn, Sn)
    plot_heatmap(posterior_distribution, f"heatmap of posterior with {num_examples} examples")
    plot_distribution_samples(Mn, Sn, num_examples, num_samples, posterior_distribution, temp_x, temp_y)




"""
Bayesian linear regression notes

Baye's theorem:
P(A|B) = (P(B|A)*P(A)) / P(B)
    - P(A|B) = posterior = P(Model | New Data)
    - P(B|A) = likelihood = P(New Data | Model)
    - P(A) = prior = P(Model)
    - P(B) = Evidence? = P(New Data)

"""
