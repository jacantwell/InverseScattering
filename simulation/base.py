from abc import ABC
import numpy as np

class Field(ABC):
    def __init__(self, E_0: float, k_0: float, propagation_angle: float):
        self.E_0 = E_0
        self.k_0 = k_0
        self.t = 1
        self.theta = propagation_angle

    # Currently lets alwasy assume the field is moving in th x direction
    def incident(self, R: np.ndarray) -> np.ndarray:
        """
        This function calculates the incident field at a given point R.

        Parameters:
        R: np.ndarray
            The point at which to calculate the incident field. Must take the form [x, y].

        Returns:
        np.ndarray
            The incident field at the point R. Takes the form [Ex, Ey].
        """
        k_vector = self.k_0 * np.array([np.cos(self.theta), np.sin(self.theta)])
        polarization_vector = np.array([-np.sin(self.theta), np.cos(self.theta)])
        E = self.E_0 * np.exp(1j * np.dot(k_vector, R) - (self.t * self.k_0))
        return E * polarization_vector
    
    def v_incident(self, R: np.ndarray) -> np.ndarray:
        """
        This function calculates the incident field at a set of points P.

        Parameters:
        P: np.ndarray
            The points at which to calculate the incident field. Must take the form [[x1, y1], [x2, y2], ...].

        Returns:
        np.ndarray
            The incident field at the points P. Takes the form [[Ex1, Ey1], [Ex2, Ey2], ...].
        """
        return np.array([self.incident(r) for r in R])
