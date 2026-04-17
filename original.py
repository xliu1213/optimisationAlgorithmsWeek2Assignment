x1_vals = np.linspace(0.0, 3.0, 400) # (I) Contour plot of f(x1,x2) with feasible region and trajectory
x2_vals = np.linspace(0.0, 4.0, 400)
X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
F_grid = (X1_grid - 1.2)**2 + 2 * (X2_grid - 2.5)**2 + 0.4 * X1_grid * X2_grid

plt.contour(X1_grid, X2_grid, F_grid, levels=25)
plt.plot(X[:, 0], X[:, 1], marker='o', markersize=3)
plt.plot(x0[0], x0[1], 'o')
plt.plot([0.5, 2.5, 2.5, 0.5, 0.5], [0.5, 0.5, 3.5, 3.5, 0.5], '--')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(0.0, 3.0)
plt.ylim(0.0, 4.0)
plt.gca().set_aspect('equal')
plt.show()

plt.plot(F) # (II) f(x_t) versus iteration
plt.xlabel('#iteration')
plt.ylabel('$f(x^{(t)})$')
plt.show()

plt.plot(X[:, 0], label='$x_1^{(t)}$') # (III) x1_t and x2_t versus iteration
plt.plot(X[:, 1], label='$x_2^{(t)}$')
plt.xlabel('#iteration')
plt.ylabel('value')
plt.legend()
plt.show()