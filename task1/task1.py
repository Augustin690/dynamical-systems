import numpy as np
import matplotlib.pyplot as plt

# 定义矩阵 Aα_1和Aα_2
def matrix_A(alpha,r):
    A=np.array([[alpha, alpha], [r, 0]])
    A_str = "\n".join(["[" + " ".join(map(str, row)) + "]" for row in A])
    return A,A_str


# 绘制相图
def plot_phase_portrait(alpha, ax,r):
    A,A_str = matrix_A(alpha,r)
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    ax.streamplot(X, Y, U, V, density=[0.5, 1])
    ax.set_title(f'A$\\alpha$=\n {A_str}',fontsize=9)


# 设置不同的 alpha 值
alphas1 = [-1, 1]
alphas2 = -1

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, alpha in enumerate(alphas1):
    plot_phase_portrait(alpha, axs[i+1],-1/4)

plot_phase_portrait(alphas2, axs[0],1/4)
plt.show()
