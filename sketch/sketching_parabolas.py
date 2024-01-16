import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

# can you create a function from 2D to 1D with a minimum at zero?
# yes, you can use a parabola

# create a 2-D vector that takes values in a grid from -1 to 1 in both axes
# and compute the function value for each point
# plot the function values in a 3-D plot

n_points = 100

u = np.linspace(-1, 1, n_points)
x, y = np.meshgrid(u, u)

X = np.vstack([x.flatten(), y.flatten()])

H = np.array([[2, -0.5], [-0.5, 0.5]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)

fig, ax = plt.subplots(ncols=3, subplot_kw={"projection": "3d"})
surf = ax[2].plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

ax[2].grid(False)
ax[2].xaxis.line.set_color('white')
ax[2].yaxis.line.set_color('white')
ax[2].zaxis.line.set_color('white')
ax[2].xaxis.pane.fill = False
ax[2].yaxis.pane.fill = False
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].set_zticklabels([])
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_zticks([])    

ax[2].set_title('Non-diagonal', fontsize=16)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

H = np.array([[2, 0], [0, 0.5]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)


surf = ax[1].plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
ax[1].grid(False)
ax[1].xaxis.line.set_color('white')
ax[1].yaxis.line.set_color('white')
ax[1].zaxis.line.set_color('white')
ax[1].xaxis.pane.fill = False
ax[1].yaxis.pane.fill = False
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_zticklabels([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_zticks([])    
ax[1].set_title('Diagonal', fontsize=16)
ax[1].set_aspect('equal')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

H = np.array([[1, 0], [0, 1]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)

surf = ax[0].plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

ax[0].grid(False)
ax[0].xaxis.line.set_color('white')
ax[0].yaxis.line.set_color('white')
ax[0].zaxis.line.set_color('white')
ax[0].xaxis.pane.fill = False
ax[0].yaxis.pane.fill = False
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].set_zticklabels([])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_zticks([])    

ax[0].set_title('Identity', fontsize=16)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

H = np.array([[2, -0.5], [-0.5, 0.5]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)

fig, ax = plt.subplots(ncols=3)
ax[2].contour(x, y, Z, cmap=cm.coolwarm)
ax[2].set_box_aspect(1)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('Non-diagonal', fontsize=16)
# Add a color bar which maps values to colors.

H = np.array([[2, 0], [0, 0.5]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)

ax[1].contour(x, y, Z, cmap=cm.coolwarm)
ax[1].set_box_aspect(1)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Diagonal', fontsize=16)

H = np.array([[1, 0], [0, 1]])

z = np.dot(np.dot(X.T, H), X)

Z = np.diag(z).reshape(n_points, n_points)

ax[0].contour(x, y, Z, cmap=cm.coolwarm)
ax[0].set_box_aspect(1)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Identity', fontsize=16)

plt.show()