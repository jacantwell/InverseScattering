import numpy as np
from simulation import simulation_factory

class DataGenerator():
    def __init__(self, simulation_config: dict):
        self.simulation = simulation_factory(simulation_config)

    def generate_grid_dataset(self, grid: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        This function generates a dataset of fields over a grid.

        Parameters:
        grid: tuple[np.ndarray, np.ndarray]
            The grid over which to generate the dataset. Must take the form (X, Y), where X and Y are 1D arrays of
            the x and y coordinates of the grid points.

        Returns:
        np.ndarray
            The dataset of fields over the grid. Takes the form [[Ex1, Ey1], [Ex2, Ey2], ...].
        """
        X, Y = grid
    
        Z = np.zeros((len(X),len(X)), dtype=complex)
    
        for i in range(0,len(X)):
            for j in range(0,len(Y)):
                R = np.array([X[i][j],Y[i][j]])
                E_x, E_y = self.simulation.field(R)
                E_mag = np.sqrt(E_x**2 + E_y**2)
                Z[i][j] = E_mag

        return Z
    
    def generate_radial_dataset(self, center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
        """
        This function generates a dataset of fields over a circle.

        Parameters:
        center: np.ndarray
            The center of the circle. Must take the form [x, y].
        radius: float
            The radius of the circle.
        num_points: int
            The number of points to sample on the circle.

        Returns:
        np.ndarray
            The dataset of fields over the circle. Takes the form [[Ex1, Ey1], [Ex2, Ey2], ...].
        """
        theta = np.linspace(0, 2*np.pi, num_points)
        X = center[0] + radius * np.cos(theta)
        Y = center[1] + radius * np.sin(theta)
    
        Z = np.zeros((len(X)), dtype=complex)
    
        for i in range(0,len(X)):
            R = np.array([X[i],Y[i]])
            E_x, E_y = self.simulation.field(R)
            E_mag = np.sqrt(E_x**2 + E_y**2)
            Z[i] = E_mag

        return Z
    
