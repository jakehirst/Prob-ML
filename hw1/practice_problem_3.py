import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import *




# log-likelihood function for a Gaussian distribution
def Gaussian_log_likelihood(params, x):
    mu, sigma = params
    return -0.5*np.sum(np.log(2*np.pi*sigma**2) + (x - mu)**2/(2*sigma**2))

#negative log-likelihood function for minimization
def Gaussian_neg_log_likelihood(params, x):
    return -Gaussian_log_likelihood(params, x)

#negative log-likelihood for the t distribution for minimization (optimizing)
def Student_t_neg_log_likelihood(params, data):
    df, loc, scale = params
    return -np.sum(t.logpdf(data, df=df, loc=loc, scale=scale))

def plot_distributions_and_data(data, noise):

    """ fit a gaussian distribtion and add it to the plot """
    #initial guesses for gaussian
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    params_init = np.array([mu_init, sigma_init])

    # Minimize negative log-likelihood using L-BFGS
    result = minimize(Gaussian_neg_log_likelihood, params_init, args=(data,), method='BFGS')

    mu_est, sigma_est = result.x

    #Plot the data and fitted Gaussian distribution
    x = np.linspace(-10, 10, 100)
    dist = norm.pdf(x, loc= result.x[0], scale= result.x[1])

    plt.hist(data, bins=np.arange(-10, 11, 0.25), density=True)
    plt.plot(x, dist, 'r-', label='Gaussian')


    """ fit a student t distribtion and add it to the plot """
    params0 = [1, 0, 2]
    bounds = [(0.01, 1000), (0, 1), (0, 3)]

    result = minimize(Student_t_neg_log_likelihood, params0, args=(data,), bounds=bounds, method='BFGS')

    df_est, mu_est, sigma_est = result.x

    dist = t.pdf(x, df_est, loc= mu_est, scale= sigma_est)

    plt.plot(x, dist, 'y-',  lw=5, alpha=0.6, label='Student t')
    plt.legend()
    if(noise):
        plt.title("Fitted Gaussian and student T distribution with noise added")
        plt.savefig("/Users/jakehirst/Desktop/Prob ML/hw1/Practice_Problem_3_with_noise")
    else:
        plt.title("Fitted Gaussian and student T distribution")
        plt.savefig("/Users/jakehirst/Desktop/Prob ML/hw1/Practice_Problem_3_no_noise")
    plt.show()
    plt.close()

mu, sigma = 0, 2 # mean and standard deviation
#generating 30 random samples from a Gaussian distribution
data = np.random.normal(mu, sigma, 30)

plot_distributions_and_data(data, noise=False)

data = np.append(data, [8,9,10])
plot_distributions_and_data(data, noise=True)
