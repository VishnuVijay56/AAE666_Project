"""
    Author: Vishnu Vijay
    Description: Class that describes a Dubins Car agent of an MAS and contains helpful functions
    Date: Created - 4/21/24
"""

#   IMPORTS: Public Library
import numpy as np
import control

#   CLASS
class Dubins2D():
    # Constructor
    def __init__(self, id, speed, initial_state, controller, dt):
        # Initialize Inputs
        self.id = id
        self.V = speed
        self.state = initial_state
        self.controller = controller
        self.dt = dt

        # Control Saturation
        self.input = 0
        min_u = -40
        max_u = 40
        self.control_sat = lambda u : min(max(min_u, u), max_u)
        
        # System Dynamics
        self.x_dot = lambda x, u : np.array([[self.V * np.cos(x.item(2))],
                                             [self.V * np.sin(x.item(2))],
                                             [u]])        
        self.step = 0


    # Iterate State of Locally Linearized Car
    def iterate_single(self, ref=None):
        if ref is None:
            ref = np.zeros(self.state.shape)
        unsat_input = self.controller( (self.state - ref), self.input )
        self.input = self.control_sat( unsat_input )
        self.state += self.x_dot(self.state, self.input)*self.dt
        self.step += 1
        return self.state
    
    # Print State
    def print(self):
        print(f"Agent {self.id} at time {self.step}")
        print(self.state.flatten())
        return