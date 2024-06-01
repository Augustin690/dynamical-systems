import numpy as np
import matplotlib.pyplot as plt

#----------------Task 1: Phase diagram under different α values--------------------
def vector_field(x, y, alpha):
    dx = alpha * x - y - x * (x ** 2 + y ** 2)
    dy = x + alpha * y - y * (x ** 2 + y ** 2)
    return dx, dy


# Draw phase diagram
def plot_phase_diagram(alpha,ax):
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    DX, DY = vector_field(X, Y, alpha)
    ax.streamplot(X, Y, DX, DY, color=np.sqrt(DX ** 2 + DY ** 2), linewidth=1, cmap='autumn')
    ax.set_title(f'α = {alpha}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True)



# Draw a phase diagram for different α values
alphas = [-0.5, 0, 0.5]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i,alpha in enumerate(alphas):
    plot_phase_diagram(alpha,axs[i])

plt.show()

#--------------Task 2: For α=1, use Euler's method to numerically compute and visualize two orbitals-----------
def euler_method(x0, y0, alpha, dt, steps):
    x = np.zeros(steps)
    y = np.zeros(steps)
    x[0], y[0] = x0, y0

    for i in range(1, steps):
        dx, dy = vector_field(x[i - 1], y[i - 1], alpha)
        x[i] = x[i - 1] + dx * dt
        y[i] = y[i - 1] + dy * dt

    return x, y


# parameter settings
alpha = 1
dt = 0.01
steps = 5000

# Initial conditions
initial_conditions = [(2, 0), (0.5, 0)]

plt.figure(figsize=(10, 8))
for x0, y0 in initial_conditions:
    x, y = euler_method(x0, y0, alpha, dt, steps)
    plt.plot(x, y, label=f'Start at ({x0}, {y0})')

plt.title(f'Trajectories at α = {alpha}')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.show()



#-------------------Task 3: Visualize the bifurcation surface of the cusp bifurcation---------------------

# Define parameter range
alpha1_values = np.linspace(-1, 1, 50)
alpha2_values = np.linspace(-1, 1, 50)
X = np.linspace(-2, 2, 200)

# Create the grid
alpha1, alpha2, x = np.meshgrid(alpha1_values, alpha2_values, X)

# Calculate the equation
F = alpha1 + alpha2 * x - x**3

# Only keep points that satisfy the equation F=0
indices = np.abs(F) < 0.05  # 允许小的误差

# Extract the corresponding alpha1, alpha2 and x
alpha1_surface = alpha1[indices]
alpha2_surface = alpha2[indices]
x_surface = x[indices]

# Draw a 3D graph
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(alpha1_surface, alpha2_surface, x_surface, c='r', marker='o')

ax.set_xlabel('$\\alpha_1$')
ax.set_ylabel('$\\alpha_2$')
ax.set_zlabel('$x$')
ax.set_title('Cusp Bifurcation Surface')

plt.show()