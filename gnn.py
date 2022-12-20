import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class GNN():
    def __init__(self, lr):
        self.H_1_1 = np.random.rand(3)
        self.H_1_2 = np.random.rand(3)

        self.H_2_1 = np.random.rand(3)
        self.H_2_2 = np.random.rand(3)

        self.intermediaires = []

        self.lr = lr

    def calc_mat(self, H, I, S, S2):
        return H[0]*I + H[1]*S + H[2]*S2

    def forwardpropagation(self, x, I, S, S2):

        a1 = self.activation(self.calc_mat(self.H_1_1, I, S, S2) @ x)
        a2 = self.activation(self.calc_mat(self.H_1_2, I, S, S2) @ x)

        b = (self.calc_mat(self.H_2_1, I, S, S2) @ a1) + \
            (self.calc_mat(self.H_2_2, I, S, S2) @ a2)

        self.intermediaires = [a1, a2, b]

        return b

    def backpropagation(self, x, y, I, S, S2):
        a1, a2, b = self.intermediaires

        dL = np.transpose(b - y)

        dH_2_1 = np.array([0, 0, 0])
        dH_2_2 = np.array([0, 0, 0])
        dH_1_2 = np.array([0, 0, 0])
        dH_1_1 = np.array([0, 0, 0])

        dH_2_1[0] = dL @ I @ a1
        dH_2_1[1] = dL @ S @ a1
        dH_2_1[2] = dL @ S2 @ a1

        dH_2_2[0] = dL @ I @ a2
        dH_2_2[1] = dL @ S @ a2
        dH_2_2[2] = dL @ S2 @ a2

        U1 = dL @ self.calc_mat(self.H_2_1, I, S, S2)
        V1 = self.calc_mat(self.H_1_1, I, S, S2) @ x

        dH_1_1[0] = U1 @ (self.activation_derivee(V1) * self.activation(I @ x))
        dH_1_1[1] = U1 @ (self.activation_derivee(V1) * self.activation(S @ x))
        dH_1_1[2] = U1 @ (self.activation_derivee(V1)
                          * self.activation(S2 @ x))

        U2 = dL @ self.calc_mat(self.H_2_2, I, S, S2)
        V2 = self.calc_mat(self.H_1_2, I, S, S2) @ x

        dH_1_2[0] = U2 @ (self.activation_derivee(V2) * self.activation(I @ x))
        dH_1_2[1] = U2 @ (self.activation_derivee(V2) * self.activation(S @ x))
        dH_1_2[2] = U2 @ (self.activation_derivee(V2)
                          * self.activation(S2 @ x))

        self.H_1_1 = self.H_1_1 - self.lr*dH_1_1
        self.H_1_2 = self.H_1_2 - self.lr*dH_1_2
        self.H_2_1 = self.H_2_1 - self.lr*dH_2_1
        self.H_2_2 = self.H_2_2 - self.lr*dH_2_2

    def activation(self, u):
        return (np.abs(u) + u)/2

    def activation_derivee(self, u):
        return (np.sign(u) + 1)/2

    def cost(self, y, ypred):
        n = len(y)
        d = ypred - y
        q = np.transpose(d)@d
        return np.trace(q)/n

    def fit(self, epochMax, S, X, Y, verbose=False):
        costLst = np.zeros(epochMax)
        mMax = len(X[0])
        n = len(X)

        I = np.eye(n)
        S2 = S@S

        for epoch in range(epochMax):
            if(verbose and (10*epoch) % epochMax == 0):
                print(f"{100*epoch/epochMax}%")

            for m in range(mMax):
                x = X[:, m].reshape(-1, 1)
                y = Y[:, m].reshape(-1, 1)

                ypred = self.forwardpropagation(x, I, S, S2)

                costLst[epoch] += self.cost(y, ypred)

                self.backpropagation(x, y, I, S, S2)
            costLst[epoch] /= mMax

        return costLst

# ""
