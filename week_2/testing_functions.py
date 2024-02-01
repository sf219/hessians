import numpy as np
import matplotlib.pyplot as plt
# import gaussian CDF
from scipy.stats import norm
# import latex rendering
from matplotlib import rc
from matplotlib import font_manager

# set latex font
plt.rcParams['text.usetex'] = True

x = np.arange(-3, 3, 0.01)

plt.figure()

beta_set = np.array([0, 0.25, 0.5])

# compute the CDF
cdf = norm.cdf(x)
# plot
plt.plot(x, x**2*cdf, label=r'$x^2\Phi(x)$', linewidth=2)

# compute the CDF
cdf = norm.cdf(x)
# plot
plt.plot(x, x**2*(cdf-0.25), label=r'$x^2(\Phi(x) - 0.25)$', linewidth=2)

# compute the CDF
cdf = norm.cdf(x)
# plot
plt.plot(x, x**2*(cdf-0.5), label=r'$x^2(\Phi(x) - 0.5)$', linewidth=2)

ax = plt.gca()
ax.xaxis.get_major_formatter()._usetex = False
ax.yaxis.get_major_formatter()._usetex = False
#remove latex rendering from ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# set labels
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$f(x)$', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()

