import numpy as np

class SIR_Lie_Transform:
    """ Implements 3rd order matrix Lie map
    """
    def __init__(self):
        self.R1=np.array([[1, 0, 0],
                          [0, 0.94176453358435, 0],
                          [0, 0.0582354664156496, 1]])

        self.R2=np.array([[0, -0.291177332078248, 0, 0, 0, 0],
                          [0, 0.282529360072762, 0, 0, 0, 0],
                          [0, 0.00864797200548554, 0, 0, 0, 0]])

        self.R3=np.array([[0, -0.0432398600274277, 0, 0.0423921193581031, 0, 0, 0, 0, 0, 0],
                          [0, 0.0423794040359606, 0, -0.0415443787011185, 0, 0, 0, 0, 0, 0],
                          [0, 0.000860455991467079, 0, -0.000847740656984653, 0, 0, 0, 0, 0, 0]])
    
    def eval(self, X):
        Xout = np.zeros_like(X)

        for i,state in enumerate(X):
            x0 = state[0]
            x1 = state[1]
            x2 = state[2]

            X1=np.array([x0,x1,x2])
            X2=np.array([x0*x0,x0*x1,x0*x2,x1*x1,x1*x2,x2*x2])
            X3=np.array([x0*x0*x0,x0*x0*x1,x0*x0*x2,x0*x1*x1,x0*x1*x2,x0*x2*x2,x1*x1*x1,x1*x1*x2,x1*x2*x2,x2*x2*x2])

            Xout[i] = np.dot(self.R1,X1)+np.dot(self.R2,X2)+np.dot(self.R3,X3)
            
        return Xout