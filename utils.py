from typing import List, Tuple, Type
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from dynamical_system import *


def plot_phase_portrait(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    X: ArrayLike,
    DX: ArrayLike,
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the phase portrait for a dynamical system with 2-d state space
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    X: ArrayLike
        Evenly spaced grid points
    DX: ArrayLike
        Velocities of X

    Returns
    -------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    """
    # TODO: add proper comments. Change code and signature and docstring if 
    # necessary. Make the plot look nicer by adding axes labels and title etc.
    ax.streamplot(X[0], X[1], DX[0], DX[1])
    ax.set_aspect("auto")
    ax.set_xlim(np.min(X[0]),np.max(X[0]))
    ax.set_ylim(np.min(X[1]),np.max(X[1]))
    return fig, ax

def plot_bifurcation_diagram(
    fig: matplotlib.figure.Figure, 
    ax: plt.Axes,
    system_class: Type[DynamicalSystem],
    dict_par_system_class: dict,
    line_color: str | None = None
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the bifurcation diagram
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    system_class: Type[DynamicalSystem]
        An object DynamicalSystem that is to be solved and plotted
    dict_par_system_class: dict
        A dictionary containing necessary all arguments to instantiate 
        system_class and to solve the dynamical system
    line_color: str | None = None
        The color of the lines in the plot

    Returns
    -------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    """
    # TODO: add proper comments. Change code and signature and docstring if 
    # necessary. Make the plot look nicer by adding axes labels and title etc.
    pars = dict_par_system_class["pars"]
    init_states = dict_par_system_class["init_states"]
    list_par_x = dict_par_system_class["list_par_x"]
    t_eval = dict_par_system_class["t_eval"]
    discrete = dict_par_system_class["discrete"]
    trajectory_matrix = np.zeros((len(pars), len(t_eval)))
    # solve the system once in each iteration with the given system 
    # parameter and the given inital state
    for idx, par in enumerate(pars):
        init_state = [init_states[idx]]
        system = system_class(par=par, list_par_x=list_par_x, discrete=discrete)
        trajectory = system.solve_system(fun=system.fun, init_state=init_state, t_eval=t_eval)
        trajectory_matrix[idx, :] = trajectory
    for idx in range(len(t_eval)):
        plt.plot(pars, trajectory_matrix[:, idx], ",", alpha=0.25, color=line_color)

    return fig, ax

def plot_3d_surface(
    X: List[np.ndarray], 
    view_angles: Tuple[float] | None = None
):
    """Visualize a 3-d surface
    
    Parameters
    ----------
    X: List[np.ndarray]
        2-d array data for each of the three axes
    view_angles: Tuple[float] | None = None
        Specify the angles to view the 3-d plot

    Returns
    -------
    None
    """
    # TODO: add proper comments. Change code and signature and docstring if 
    # necessary. Make the plot look nicer by adding axes labels and title etc.
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[0], X[1], X[2], cmap='viridis')
    if view_angles is not None:
        ax.view_init(view_angles[0], view_angles[1], view_angles[2])

def plot_3d_traj(trajectories: List[np.ndarray]):
    """Visualize the 3-d trajectory
    
    Parameters:
    -----------
    trajectories: List[np.ndarray]
        1-d array data for each of the three axes
    
    Returns
    -------
    None
    """
    # TODO: add proper comments. Change code and signature and docstring if 
    # necessary. Make the plot look nicer by adding axes labels and title etc.
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    for trajectory in trajectories:
        ax.plot(trajectory[0], trajectory[1], trajectory[2], linewidth=0.1)