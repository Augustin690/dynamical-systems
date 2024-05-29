import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


#----------任务1：r从0到4的分岔图
# 参数设置
r_values = np.linspace(0, 4, 1000)
iterations = 1000##迭代次数
last = 100##绘制次数

# 初始化数组来存储结果
x = 1e-5 * np.ones(len(r_values))

# 绘制分岔图
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(iterations):
    x = r_values * x * (1 - x)
    # 只绘制最后的点，以避开初始的转态
    if i >= (iterations - last):
        ax.plot(r_values, x, c='b', alpha=0.25)

ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
ax.set_title('Bifurcation diagram of the logistic map')
ax.set_xlabel('r')
ax.set_ylabel('x')
plt.grid(True)
plt.show()






#----------任务2：可视化单一轨迹
# Lorenz系统方程
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 参数设置
sigma = 10
beta = 8/3
rho = 28
initial_state = [10, 10, 10]
t_end = 1000
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 10000)

# 计算轨迹
sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, beta, rho), t_eval=t_eval)

# 绘制轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_title('Lorenz Attractor with initial condition (10, 10, 10)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#----------------任务3: 测试对初始条件的敏感性------------
# 另一初始条件
initial_state_perturbed = [10 + 1e-8, 10, 10]

# 计算另一条轨迹
sol_perturbed = solve_ivp(lorenz, t_span, initial_state_perturbed, args=(sigma, beta, rho), t_eval=t_eval)

# 计算两条轨迹之间的差异
diff = np.sqrt((sol.y[0] - sol_perturbed.y[0])**2 +
               (sol.y[1] - sol_perturbed.y[1])**2 +
               (sol.y[2] - sol_perturbed.y[2])**2)

# 绘制两条轨迹之间的差异
plt.figure(figsize=(10, 6))
plt.plot(t_eval, diff)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between trajectories with initial conditions (10, 10, 10) and (10 + 1e-8, 10, 10)')
plt.grid(True)
plt.show()


#------------任务4：更改参数 ρ 为 0.5，并再次计算轨迹----------------
# 更改参数
rho_new = 0.5

# 计算轨迹
sol_new = solve_ivp(lorenz, t_span, initial_state, args=(sigma, beta, rho_new), t_eval=t_eval)
sol_perturbed_new = solve_ivp(lorenz, t_span, initial_state_perturbed, args=(sigma, beta, rho_new), t_eval=t_eval)

# 计算两条轨迹之间的差异
diff_new = np.sqrt((sol_new.y[0] - sol_perturbed_new.y[0])**2 +
                   (sol_new.y[1] - sol_perturbed_new.y[1])**2 +
                   (sol_new.y[2] - sol_perturbed_new.y[2])**2)

# 绘制轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_new.y[0], sol_new.y[1], sol_new.y[2], lw=2.0,c='r')
ax.set_title('Lorenz Attractor with initial condition (10, 10, 10) and ρ = 0.5')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 绘制两条轨迹之间的差异
plt.figure(figsize=(10, 6))
plt.plot(t_eval, diff_new)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between trajectories with initial conditions (10, 10, 10) and (10 + 1e-8, 10, 10) with ρ = 0.5')
plt.grid(True)
plt.show()
