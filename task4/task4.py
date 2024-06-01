import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


#----------Task 1: Bifurcation diagram of r from 0 to 4
# Parameter setting
r_values = np.linspace(0, 4, 1000)
iterations = 1000##Times of iterations
last = 100##Draw times

# Initialize an array to store the results
x = 1e-5 * np.ones(len(r_values))

# Draw a bifurcation diagram
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(iterations):
    x = r_values * x * (1 - x)
    # Draw only the last point to avoid the initial transition
    if i >= (iterations - last):
        ax.plot(r_values, x, c='b', alpha=0.25)

ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
ax.set_title('Bifurcation diagram of the logistic map')
ax.set_xlabel('r')
ax.set_ylabel('x')
plt.grid(True)
plt.show()






#----------Task 2: Visualize a single trajectory
# Lorenz system equation
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# parameter settings
sigma = 10
beta = 8/3
rho = 28
initial_state = [10, 10, 10]
t_end = 1000
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 10000)

# Calculate trajectory
sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, beta, rho), t_eval=t_eval)

# Draw the trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_title('Lorenz Attractor with initial condition (10, 10, 10)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#----------------Task 3: Testing sensitivity to initial conditions------------
# Another initial condition
initial_state_perturbed = [10 + 1e-8, 10, 10]

# Calculate another trajectory
sol_perturbed = solve_ivp(lorenz, t_span, initial_state_perturbed, args=(sigma, beta, rho), t_eval=t_eval)

# Calculate the difference between the two trajectories
diff = np.sqrt((sol.y[0] - sol_perturbed.y[0])**2 +
               (sol.y[1] - sol_perturbed.y[1])**2 +
               (sol.y[2] - sol_perturbed.y[2])**2)

# Plot the difference between the two trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_eval, diff)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between trajectories with initial conditions (10, 10, 10) and (10 + 1e-8, 10, 10)')
plt.grid(True)
plt.show()


#------------Task 4: Change parameter ρ to 0.5 and calculate the trajectory again----------------
# Change parameters
rho_new = 0.5

# Calculate trajectory
sol_new = solve_ivp(lorenz, t_span, initial_state, args=(sigma, beta, rho_new), t_eval=t_eval)
sol_perturbed_new = solve_ivp(lorenz, t_span, initial_state_perturbed, args=(sigma, beta, rho_new), t_eval=t_eval)

# Calculate the difference between the two trajectories
diff_new = np.sqrt((sol_new.y[0] - sol_perturbed_new.y[0])**2 +
                   (sol_new.y[1] - sol_perturbed_new.y[1])**2 +
                   (sol_new.y[2] - sol_perturbed_new.y[2])**2)

# Draw the trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_new.y[0], sol_new.y[1], sol_new.y[2], lw=2.0,c='r')
ax.set_title('Lorenz Attractor with initial condition (10, 10, 10) and ρ = 0.5')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Plot the difference between the two trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_eval, diff_new)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between trajectories with initial conditions (10, 10, 10) and (10 + 1e-8, 10, 10) with ρ = 0.5')
plt.grid(True)
plt.show()
