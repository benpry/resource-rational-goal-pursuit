"""
This file contains the code for the simulated microworld [expand]
"""

import torch
import numpy as np
from mpmath import *


class Microworld:
    """
    Represents a simulated micro-world (SMW) with endogenous variables and exogenous inputs.
    """
    def __init__(self, A=None, B=None, init=None, von_mises_parameter=None, exponential_parameter=None):
        """
        A: transition matrix for endogenous variables
        B: transition matrix for exogenous variables
        init: the initial endogenous state of the system
        von_mises_parameter: the parameter for the exponential distribution (length noise)
        exponential_parameter: the concentration parameter for the von mises distribution (radial noise)
        """
        # initialize the endogenous state, making sure it is a tensor
        if type(init) != torch.Tensor:
            self.endogenous_state = torch.tensor(init, dtype=torch.float64)
        else:
            self.endogenous_state = init

        # initialize the transition matrices
        self.A = A  # matrix endogenous_n x endogenous_n
        self.B = B  # matrix endogenous_n x exogenous_n

        # initialize the von mises and exponential parameters
        self.von_mises_parameter = von_mises_parameter
        self.exponential_parameter = exponential_parameter

    def step(self, action):
        """Updates the endogenous state by applying an action without noise."""
        self.endogenous_state = torch.matmul(self.A, self.endogenous_state.t()).t() + torch.matmul(self.B, action.t())\
            .t()

    def step_with_model(self, action, noise=True):
        """
        Updates the endogenous state by applying an action, possibly with noise if noise is set to true
        """
        # update endogenous state
        self.endogenous_state = self.A.mv(self.endogenous_state) + self.B.mv(action)

        if not noise:
            return

        # print('before', self.endogenous_state)
        angles, length = self.to_spherical(self.endogenous_state.numpy())

        # sample length and radial noise from exponential and von mises distributions
        length_noise = np.random.exponential(self.exponential_parameter)
        angle_noise = np.random.vonmises(0, self.von_mises_parameter, 4)

        # add noise to angles and length
        angles += angle_noise
        length += length_noise

        # turn spherical coordinates back to Cartesian coordinates and update endogenous state
        self.endogenous_state = torch.tensor(self.to_endogenous(length, angles), dtype=torch.float64)

    def arccot(self, x):
        """
        compute arc cotangent of x, adding pi if result is nonpositive.
        """
        if x > 0:
            return acot(x)
        else:
            return np.pi + acot(x)

    def to_spherical(self, vec):
        """
        Transform a vector of 5 dimensions to spherical coordinates

        vec: numpy array of len 5 reprenting a visited state
        """

        r = np.linalg.norm(vec)
        if np.linalg.norm(np.delete(vec, [0])) == 0.:
            if vec[0] >= 0:
                phi_1 = 0.
            else:
                phi_1 = np.pi
        else:
            phi_1 = float(self.arccot(vec[0] / np.sqrt(np.sum(np.delete(vec, [0])**2))))

        if np.linalg.norm(np.delete(vec, [0, 1])) == 0.:
            if vec[1] >= 0:
                phi_2 = 0.
            else:
                phi_2 = np.pi
        else:
            phi_2 = float(self.arccot(vec[1] / np.sqrt(np.sum(np.delete(vec, [0, 1])**2))))

        if np.linalg.norm(np.delete(vec, [0, 1, 2])) == 0.:
            if vec[2] >= 0:
                phi_3 = 0.
            else:
                phi_3 = np.pi
        else:
            phi_3 = float(self.arccot(vec[2] / np.sqrt(np.sum(np.delete(vec, [0, 1, 2])**2))))

        if vec[4] == 0.:
            if vec[3] >= 0.:
                phi_4 = 0.
            else:
                phi_4 = np.pi
        else:
            phi_4 = 2 * float(self.arccot((vec[3] + np.sqrt(np.sum(np.delete(vec, [0, 1, 2])**2))) / vec[4]))
        return np.array([phi_1, phi_2, phi_3, phi_4]), r

    def to_endogenous(self, r, angles):
        """
        Transform radian,angles to endogenous state

        r: radian of a state
        angles: angles of a state
        """

        x_1 = r * np.cos(angles[0])
        x_2 = r * np.sin(angles[0]) * np.cos(angles[1])
        x_3 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
        x_4 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.cos(angles[3])
        x_5 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3])

        return np.array([x_1, x_2, x_3, x_4, x_5])
