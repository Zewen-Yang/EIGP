import numpy as np
import math
from utils.common import * 
from scipy.stats import norm
from utils.GPmodel import GPmodel


class AEIGP():
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

    def addData_EIGP_agent(self, i_agent, x, y):
        self.agents[i_agent].addData_EIGP(x, y)
        self.mas_dataQuantity += 1
    
    def agentsSelect_aEIGP(self, i_agent, x):
        if self.mas_dataQuantity == 0:
            self.requestNieghborsList[i_agent] = self.neighbors_list[i_agent]
            self.aggWieghtsList[i_agent] = equalProportions(len(self.neighbors_list[i_agent]))
        else:
            i_agent_epsilonList = []
            temp_neighbors = self.neighbors_list[i_agent]
            for s_neighbor in temp_neighbors:
                # self.agents[s_neighbor].prePredict_EIGP_fullData(x)
                self.agents[s_neighbor].prePredict_EIGP_allocate(x)
                i_agent_epsilonList.append(self.agents[s_neighbor].epsilon)

            i_agent_trustArray = np.array(i_agent_epsilonList)
            max_value = max(i_agent_trustArray)
            std_value = np.sqrt(np.var(i_agent_trustArray))
            # indices =  np.where(np.array(i_agent_trustArray) >=0)
            indices =  np.where(np.array(i_agent_trustArray) > max_value - 2*std_value)
            corresponding_agents = np.array(temp_neighbors)[indices]
            self.requestNieghborsList[i_agent] = corresponding_agents.astype(int).tolist()
            corresponding_trust = i_agent_trustArray[indices].tolist()
            self.aggWieghtsList[i_agent] = norm(loc=max_value, scale=std_value).pdf(corresponding_trust).tolist()
    
    def predict_aEIGP(self, i_agent, x):
        if self.mas_dataQuantity == 0:
            mu = np.zeros((self.y_dim,1))
            var = self.sigmaF ** 2 * np.ones([self.y_dim, 1], dtype=float)
            return np.squeeze(mu), var
        else:
            self.agentsSelect_aEIGP(i_agent, x)
            temp_agents_list = self.requestNieghborsList[i_agent]
            self.requestNeighborQuantity_list[i_agent].append(len(temp_agents_list))
            temp_weights = self.aggWieghtsList[i_agent]
            temp_weights = getProportions(temp_weights)
            temp_mu_list = []
            temp_var_list = []
            weight_list = []
            weight_beta_k_list = []

            for i, s_agent in enumerate(temp_agents_list):
                # mu, var = self.agents[s_agent].predict(x)
                mu, var = self.agents[s_agent].predict_allocate(x)
                weight = temp_weights[i]
                weight_list.append(np.squeeze(weight))
                beta_k = math.log((self.sigmaF**2 + self.sigmaN**2)/var) 
                temp_mu_list.append(np.squeeze(mu))
                temp_var_list.append(np.squeeze((weight**(1/2)) * (beta_k/var)**(1/2)))
                weight_beta_k_list.append(np.squeeze((weight**(1/2)) * (beta_k**(1/2))))

                # weight_beta_k_list.append(np.squeeze(beta_k))
                # temp_var_list.append(np.squeeze(1*beta_k/(var)))
            bcm_var = (1-sum(weight_beta_k_list))/(self.sigmaF**2+ self.sigmaN**2)
            var_AEIGP = np.sum(temp_var_list )+ bcm_var.ravel()
            weights = temp_var_list/var_AEIGP
            mu_AEIGP = np.dot(temp_mu_list, weights)
            return mu_AEIGP, var_AEIGP
