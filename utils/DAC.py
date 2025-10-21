import numpy as np
from utils.common import * 
from scipy.integrate import solve_ivp
from utils.GPmodel import GPmodel


class DAC():
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
    
    def predict_DAC_IND(self, i_agent, x):
        self.agents[i_agent].prePredict(x)
        mu, var = self.agents[i_agent].predict(x)
        return mu, var
    
    def predict_DAC(self, p0, MAS_Ind_mu_sig, A):
        t_span = (0, 0.1)
        mu_DAC = [[] for _ in range(self.agentQuantity)]
        r = MAS_Ind_mu_sig
        ADCstate_dim = 2*self.y_dim
        # Solve the ODE using solve_ivp
        sol = solve_ivp(Pdyn, t_span, p0, args=(r, ADCstate_dim, A))
        Pstates = sol.y[:,-1]

        Xi = (MAS_Ind_mu_sig.ravel() - Pstates)/self.agentQuantity
        SigAve = 1./Xi[np.arange(1, 2 * self.agentQuantity+1, ADCstate_dim, dtype=int)]
        MuAve = SigAve*Xi[np.arange(0, 2 * self.agentQuantity-1, ADCstate_dim, dtype=int)]
        [mu_DAC[i_agent].append(MuAve[i_agent]) for i_agent in range(self.agentQuantity)]

        return mu_DAC, Pstates
    