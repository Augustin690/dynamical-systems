from typing import List, Callable
from numpy.typing import ArrayLike
import numpy as np

class DynamicalSystem:
    """This class defines a dynamical system
    
    Methods
    --------
    solve_system(fun: Callable, init_state: ArrayLike, t_eval: ArrayLike):
        Solve the dynamical system
    """
    def __init__(self, discrete: bool = False):
        """Parameters
        -------------
        discrete: bool = False
            If true, the dynamical system is time-discrete
        """
        self.discrete = discrete

    def solve_system(
        self,
        fun: Callable,
        init_state: ArrayLike,
        t_eval: ArrayLike
    ) -> ArrayLike:
        """Solve the dynamical system

        Given the evolution rules, the initial point, and the time steps, we 
        obtain the trajectory of the point. The solving method is different 
        for time-discrete system, so two methods are implemented here. 

        Parameters
        ----------
        fun: Callable
            Evolution operator
        init_state: ArrayLike
            Initial state of the system
        t_eval: ArrayLike
            Time steps of the trajectory

        Returns
        -------
        trajectory: ArrayLike
            Trajectory of the inital point in time
        """
        # TODO: implement the solver. You can change the signature. If so, please 
        # don't forget to change the docstring.
        if not self.discrete:
            pass
        else:
            pass
        return 0
    
    def _set_grid_coordinates(self, list_par_x: List[List[int]]) -> List[np.ndarray]:
        
        """Set up the coordinates. For multidimensional cases use meshgrid"""
        match len(list_par_x):
            case 1:
                return np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2])
            case 2:
                X1, X2 = np.meshgrid(
                    np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                    np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2])
                )
                return [X1, X2]
            case 3:
                X1, X2, X3 = np.meshgrid(
                    np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                    np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2]),
                    np.linspace(list_par_x[2][0], list_par_x[2][1], list_par_x[2][2])
                )
                return [X1, X2, X3]


class Task1(DynamicalSystem):
    """TODO: write a docstring"""
    def __init__(self, par, list_par_x, *args, **kwargs):
        """TODO: write a docstring"""
        super().__init__(*args, **kwargs)
        self.matrix = par
        self.X = self._set_grid_coordinates(list_par_x)

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """TODO: write a docstring"""

        # TODO: implement the evolution operator. You can change the signature.

        return 0
    

# TODO: write classes for other tasks