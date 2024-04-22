"""
    Author: Vishnu Vijay
    Description: Class that describes a Dubins Car agent of an MAS and contains helpful functions
    
"""

#   IMPORTS: Public Library
import numpy as np
import control

#   CLASS
class Dubins2D():
    # Constructor
    def __init__(self, id, initial_state, dt):
        # Initialize Inputs
        self.id = id
        self.state = initial_state
        self.dt = dt
        
        # System Dynamics
        self.V = 1
        self.x_dot = lambda x, u : np.array([[self.V * np.cos((x.flatten())[2])],
                                             [self.V * np.sin((x.flatten())[2])],
                                             [u]])
        
        self.recompute_A()
        self.B = np.array([[0],
                           [0],
                           [1]])
        
        self.step = 0
        
        # State and Input Cost Matrices
        self.Q = np.diag([100, 100, 10])
        self.R = np.diag([5])
        
        # Compute control gain
        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)


    # Iterate State of Locally Linearized Car
    def iterate_single(self, ref):
        if self.step % 5 == 0:
            self.recompute_A()
            self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)
        u = -self.K @ (self.state - ref)
        self.state = self.x_dot(self.state, u.item(0))
        self.step += 1
        return self.state
    
    # Recompute the system matrix A
    def recompute_A(self):
        self.A = np.array([[0, 0, -self.V*np.sin( self.state.item(2) )],
                           [0, 0, self.V*np.cos( self.state.item(2) )],
                           [0, 0, 0]])
    
    # Set Neighbors
    def set_neighbors(self, nbr_list):
        self.nbr_list = nbr_list
        self.num_nbr = len(nbr_list)
        return None
    
    # Print State
    def print(self):
        print("Agent ", self.id)
        print(self.state.flatten())
        return