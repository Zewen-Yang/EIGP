import math
import numpy as np
from utils.common import * 
from utils.GPmodel import GPmodel


class GPOE():
    def __init__(self, x_dim, y_dim, indivDataThersh, 
                 sigmaN, sigmaF, sigmaL, 
                 priorFuncList, agentQuantity, 
                 Graph):
        
        self.indivDataThersh = indivDataThersh
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.sigmaN = sigmaN 
        self.sigmaF = sigmaF
        self.sigmaL = sigmaL
        self.priorFunc = priorFuncList
        self.agentQuantity = agentQuantity
        self.agents = []
        self.mas_dataQuantity = 0
        self.G = Graph

        self.error_list = [[[] for _ in range(agentQuantity)] for _ in range(agentQuantity)]
        self.neighbors_list = [[] for _ in range(agentQuantity)]
        self.requestNieghborsList = [[] for _ in range(agentQuantity)]
        self.requestNeighborQuantity_list = [[] for _ in range(agentQuantity)]
        self.aggWieghtsList = [[] for _ in range(agentQuantity)]
        

        for i in range(agentQuantity):
            agent = GPmodel(x_dim, y_dim, indivDataThersh, sigmaN, sigmaF, sigmaL, priorFuncList[i])
            self.agents.append(agent)
            self.neighbors_list[i] = list(self.G.neighbors(i))


    def addDataEntire_agents(self, X_train_list, Y_train_list):
        for i_agent in range(self.agentQuantity):
            self.agents[i_agent].addDataEntire(X_train_list[i_agent], Y_train_list[i_agent])
            self.mas_dataQuantity += Y_train_list[i_agent].shape[1]


    def addData_agent(self, i_agent, x, y):
        self.agents[i_agent].addData(x, y)
        self.mas_dataQuantity += 1
    
    
    def predict_GPOE(self, i_agent, x):
        temp_agents_list = self.neighbors_list[i_agent]
        temp_mu_list = []
        temp_var_list = []
        for s_agent in temp_agents_list:
            self.agents[s_agent].prePredict(x)
            mu, var = self.agents[s_agent].predict(x)
            beta_k = math.log((self.sigmaF**2 + 1e-5)/var) 
            temp_mu_list.append(np.squeeze(mu))
            temp_var_list.append(np.squeeze(beta_k/(var)))
        var_GPOE = np.sum(temp_var_list)
        weights_g = temp_var_list/var_GPOE
        mu_GPOE = np.dot(temp_mu_list, weights_g)
        return mu_GPOE, var_GPOE
