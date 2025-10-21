import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from typing import Tuple
from utils.common import * 



class GPmodel():
    def __init__(self, x_dim, y_dim, indivDataThersh, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFunc):
        self.dataQuantity = 0
        self.indivDataThersh = indivDataThersh  # data limit per local GP
        self.x_dim = x_dim  # dimensionality of X
        self.y_dim = y_dim  # dimensionality of Y
        self.sigmaN = sigmaN 
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL
        self.priorFunc = priorFunc

        self.X = np.full([x_dim, indivDataThersh], np.inf)
        self.X_all = np.zeros([x_dim, 1])
        self.Y = np.full([y_dim, indivDataThersh], np.inf)
        self.Y_all = np.zeros([y_dim, ])
        self.K = np.full([indivDataThersh, indivDataThersh], np.inf)  # covariance matrix
        self.K_border = np.full([1, indivDataThersh], np.inf)
        self.K_corner = None
        self.rho = None

        self.alpha = [[] for _ in range(y_dim)]  # prediction vectors
        self.predictError = None
        self.epsilon = 0

        self.K_allocate = None
        self.K_border_allocate = None
        self.Y_allocate = None
        self.X_allocate= None
        self.alpha_allocate = [[] for _ in range(y_dim)]


    def kernel(self, Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
        kernlMatrix = np.zeros([Xi.shape[1], Xj.shape[1]], dtype=float)
        for Xj_Nr in range(Xj.shape[1]):
            kernlMatrix[:, Xj_Nr] = (self.sigmaF ** 2) * np.exp(
                -0.5 * np.sum(
                    ((Xi - np.matlib.repmat(Xj[:,Xj_Nr].reshape(self.x_dim, -1), 1, Xi.shape[1])) / self.sigmaL
                    ) ** 2, axis=0))  # axis = 0 means sum all rows (all dimensions)
        return kernlMatrix
        
    def addDataEntire(self, X_in, Y_in):
        AlldataQuantity = min(X_in.shape[1], self.indivDataThersh)
        self.dataQuantity = AlldataQuantity
        self.X[:, range(self.dataQuantity)] = X_in[:, range(AlldataQuantity)]
        self.Y[:, range(self.dataQuantity)] = Y_in[:, range(AlldataQuantity)]
        self.K[0:self.dataQuantity, 0:self.dataQuantity] \
            = self.kernel(self.X[:,range(self.dataQuantity)], self.X[:,range(self.dataQuantity)])\
                  + self.sigmaN ** 2 * np.eye(self.dataQuantity, dtype=float)

    def addData(self, x, y):
        self.X_all = np.append(self.X_all, x, axis=1)
        self.Y_all = np.append(self.Y_all, y, axis=0)
        if self.dataQuantity==0:
            self.X[:, 0] = x.ravel()
            self.Y[:, 0] = y
            self.K[0, 0] = self.sigmaF**2 + self.sigmaN ** 2
            self.dataQuantity = self.dataQuantity + 1
        elif self.dataQuantity < self.indivDataThersh:
            self.X[:, self.dataQuantity] = x.ravel()
            self.Y[:, self.dataQuantity] = y
            temp_K_boder = self.kernel(self.X[:, 0:self.dataQuantity], x).ravel()
            self.K[self.dataQuantity, 0:self.dataQuantity] = temp_K_boder
            self.K[0:self.dataQuantity, self.dataQuantity] = temp_K_boder
            self.K[self.dataQuantity, self.dataQuantity] = self.sigmaF**2 + self.sigmaN ** 2
            self.dataQuantity = self.dataQuantity + 1
        else:
            self.X[:, 0:self.dataQuantity-1] = self.X[:, 1: self.dataQuantity]
            self.X[:, self.dataQuantity-1] = x.ravel()
            self.Y[:, 0:self.dataQuantity-1] = self.Y[:, 1: self.dataQuantity]
            self.Y[:, self.dataQuantity-1] = y
            temp_K_boder = self.kernel(self.X[:, 0:self.dataQuantity-1], x).ravel()
            self.K[0: self.dataQuantity-1, 0:self.dataQuantity-1] = self.K[1 : self.dataQuantity, 1:self.dataQuantity]
            self.K[self.dataQuantity-1, 0:self.dataQuantity-1] = temp_K_boder
            self.K[0:self.dataQuantity-1, self.dataQuantity-1] = temp_K_boder
            self.K[self.dataQuantity-1, self.dataQuantity-1] = self.sigmaF**2 + self.sigmaN ** 2

    def addData_EIGP(self, x, y):
        if self.dataQuantity==0:
            self.X[:, 0] = x.ravel()
            self.Y[:, 0] = y
            self.K[0, 0] = self.sigmaF**2 + self.sigmaN ** 2
            self.dataQuantity = self.dataQuantity + 1
        elif self.dataQuantity < self.indivDataThersh:
            self.X[:, self.dataQuantity] = x.ravel()
            self.Y[:, self.dataQuantity] = y
            temp_K_boder = self.kernel(self.X[:, 0:self.dataQuantity], x).ravel()
            self.K[self.dataQuantity, 0:self.dataQuantity] = temp_K_boder
            self.K[0:self.dataQuantity, self.dataQuantity] = temp_K_boder
            self.K[self.dataQuantity, self.dataQuantity] = self.sigmaF**2 + self.sigmaN ** 2
            self.dataQuantity = self.dataQuantity + 1
        else:
            delet_index = np.argmin(self.kernel(self.X[:, 0:self.dataQuantity], x))
            self.X[:, delet_index] = x.ravel()
            self.Y[:, delet_index] = y
            temp_K_vector = self.kernel(self.X[:, 0: self.dataQuantity], x)
            self.K[delet_index, :] = temp_K_vector.ravel()
            self.K[:, delet_index] = temp_K_vector.ravel()
            self.K[delet_index, delet_index] += self.sigmaN ** 2

    def prePredict(self,x):
        if self.dataQuantity == 0:
            self.trustValue = 0
        else:
            temp_k_vector = self.kernel(self.X[:, 0:self.dataQuantity], x)
            self.K_border[0, 0:self.dataQuantity] = temp_k_vector.ravel()
            for i in range(self.y_dim):
                    self.alpha[i] = np.linalg.solve(self.K[0:self.dataQuantity, 0:self.dataQuantity], 
                                                    (self.Y[i, 0:self.dataQuantity] - self.priorFunc(self.X[:, 0: self.dataQuantity]).flatten())
                                                    ).transpose()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dataQuantity == 0:
            mu = self.priorFunc(self.X[:, 0: self.dataQuantity])
            var = self.sigmaF ** 2 * np.ones([self.y_dim, 1], dtype=float)
            return mu, var
        else:
            mu = np.zeros([self.y_dim, 1], dtype=float)
            var = np.zeros([self.y_dim, 1], dtype=float)
            temp_K_border = self.K_border[0, 0:self.dataQuantity]
            temp_K = self.K[0 : self.dataQuantity, 0:self.dataQuantity]
            for i_dim in range(self.y_dim):
                mu[[i_dim], :] = self.priorFunc(x) +  np.dot(self.alpha[i_dim], temp_K_border)
                temp_KinvKborder = np.linalg.solve(temp_K, temp_K_border)
                var[[i_dim], :] = self.kernel(x,x) - np.dot(temp_K_border.transpose(), temp_KinvKborder)
            return mu, var
    
    def prePredict_EIGP_allocate(self,x):
        if self.dataQuantity == 0:
            pass
        else:
            temp_k_vector = self.kernel(self.X[:, 0:self.dataQuantity], x)
            self.K_border[0, 0:self.dataQuantity] = temp_k_vector.ravel()
            # define rho
            # self.rho = np.median(temp_k_vector)
            self.rho = np.mean(temp_k_vector)
            indices = np.where(temp_k_vector >= self.rho)[0]

            temp_K = self.K[0:self.dataQuantity, 0:self.dataQuantity]
            K_allocate = temp_K[:, indices]
            K_allocate = K_allocate[indices, :]
            self.K_allocate = K_allocate
            self.K_border_allocate = temp_k_vector[indices]
            self.Y_allocate = self.Y[:,indices]
            self.X_allocate= self.X[:,indices]

            for i in range(self.y_dim):
                    temp_KinvYtrain_allocate= np.linalg.solve(K_allocate, 
                                                    (self.Y_allocate[i,:] - self.priorFunc(self.X_allocate).flatten()))
                    self.alpha_allocate[i] = temp_KinvYtrain_allocate.transpose()

            self.predictError = np.absolute(-self.sigmaN**2 * self.alpha_allocate[0])
            epsilon_temp = np.multiply(self.predictError, self.K_border_allocate.transpose())
            self.epsilon = np.sum(epsilon_temp)/len(epsilon_temp)
   
    def predict_mu(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dataQuantity == 0:
            mu = self.priorFunc(self.X[:, 0: self.dataQuantity])
            return mu
        else:
            mu = np.zeros([self.y_dim, 1], dtype=float)
            temp_K_border = self.K_border[0, 0:self.dataQuantity]
            for i_dim in range(self.y_dim):
                mu[[i_dim], :] = self.priorFunc(x) +  np.dot(self.alpha[i_dim], temp_K_border)
            return mu
        
    def predict_allocate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dataQuantity == 0:
            mu = self.priorFunc(self.X[:, 0: self.dataQuantity])
            var = self.sigmaF ** 2 * np.ones([self.y_dim, 1], dtype=float)
            return mu, var
        else:
            mu = np.zeros([self.y_dim, 1], dtype=float)
            var = np.zeros([self.y_dim, 1], dtype=float)
            temp_K_border = self.K_border_allocate
            temp_K = self.K_allocate
            for i_dim in range(self.y_dim):
                mu[[i_dim], :] = self.priorFunc(x) +  np.dot(self.alpha_allocate[i_dim], temp_K_border)
                temp_KinvKborder = np.linalg.solve(temp_K, temp_K_border)
                var[[i_dim], :] = self.kernel(x,x) - np.dot(temp_K_border.transpose(), temp_KinvKborder)
            return mu, var
        
    def predict_mu_allocate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dataQuantity == 0:
            mu = self.priorFunc(self.X[:, 0: self.dataQuantity])
            return mu
        else:
            mu = np.zeros([self.y_dim, 1], dtype=float)
            temp_K_border = self.K_border_allocate
            for i_dim in range(self.y_dim):
                mu[[i_dim], :] = self.priorFunc(x) +  np.dot(self.alpha_allocate[i_dim], temp_K_border)
            return mu



    