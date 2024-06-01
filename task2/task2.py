import numpy as np
import matplotlib.pyplot as plt

# Define the range of alpha
alpha_values1 = np.linspace(-1, 1, 400)
alpha_values2 = np.linspace(0, 6, 400)

# Calculated equilibrium point
x_pos1 = np.sqrt(alpha_values1[alpha_values1>= 0])
x_neg1 = -np.sqrt(alpha_values1[alpha_values1 >= 0])

x_pos2 = np.sqrt((alpha_values2 - 3) / 2)
x_neg2 = -np.sqrt((alpha_values2 - 3) / 2)

# Draw a bifurcation diagram
plt.figure(figsize=(10, 6))
plt.plot(alpha_values1[alpha_values1 >= 0], x_pos1, 'b-', label='Stable equilibrium')
plt.plot(alpha_values1[alpha_values1 >= 0], x_neg1, 'r--', label='Unstable equilibrium')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel('α')
plt.ylabel('x')
plt.title('Bifurcation Diagram of the System: ẋ = α - x²')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(alpha_values2[alpha_values2 >= 3], x_pos2[alpha_values2 >= 3], 'b-', label='Stable equilibrium')
plt.plot(alpha_values2[alpha_values2 >= 3], x_neg2[alpha_values2 >= 3], 'r--', label='Unstable equilibrium')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(3, color='gray', linewidth=0.5, linestyle='dotted')
plt.xlabel('α')
plt.ylabel('x')
plt.title('Bifurcation Diagram of the System: ẋ = α - 2x² - 3')
plt.legend()
plt.grid(True)
plt.show()
