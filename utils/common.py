import numpy as np

def minmaxScaling(values, _new_min = 0, _new_max =1):
    # values = [0, 0.0001, 10000]
    # Calculate the minimum and maximum values in the list
    min_value = min(values)
    max_value = max(values)
    # Define the new range for scaling
    new_min = _new_min
    new_max = _new_max
    # Initialize an empty list to store the scaled values
    scaled_values = []
    # Iterate through the original values and scale them
    for value in values:
        scaled_value = (value - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
        scaled_values.append(scaled_value)
    return scaled_values


def getProportions(values):
        # Calculate the sum of all elements in the list
        total = sum(values)
        # Calculate the proportions by dividing each element by the total
        proportions = [value / total for value in values]
        return proportions


def equalProportions(n):
    if n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")
    # Calculate the proportion value
    proportion_value = 1 / n
    # Create a list with n elements, each set to the proportion_value
    proportions = [proportion_value] * n
    return proportions


def ndgridj(grid_min, grid_max, ns):
    D = len(ns)

    grid_max = np.array(grid_max)
    grid_min = np.array(grid_min)
    ns = np.array(ns)

    if grid_max.shape != (D,) or grid_min.shape != (D,):
        raise ValueError("grid_max, grid_min, and ns must have the same dimensions")

    if ns.ndim != 1:
        raise ValueError("grid_max, grid_min, and ns must be 1D arrays")

    if np.any(grid_max - grid_min < 0):
        raise ValueError("grid_max is not always larger than grid_min")

    gg = [np.linspace(grid_min[i], grid_max[i], ns[i]) for i in range(D)]

    grid = np.meshgrid(*gg, indexing='ij')
    grid = np.array(grid).reshape(D, -1).T

    max_dist = None
    if D > 1:
        dist = (grid_max - grid_min) / ns
        max_dist = np.sqrt(np.sum(dist ** 2)) / 2

    return grid.transpose(), max_dist

def Pdyn(t, y, r, ADCstate_dim, A_Mat):
    p=y
    dpdt = np.empty_like(p)
    agentQuantity = A_Mat.shape[0]

    for i_agent in range(agentQuantity):
        slicing = slice(i_agent*ADCstate_dim, (i_agent+1)*ADCstate_dim)
        i_agent_Pstate = p[slicing]
        i_agent_Refstate = r[slicing]
        r_consen_temp = np.kron(1e2*A_Mat[i_agent,:], np.ones([1,ADCstate_dim])).transpose() \
                        * (np.kron(np.ones([agentQuantity,1]), i_agent_Refstate) - r)
        i_agent_r_consen = np.sum(r_consen_temp.reshape(ADCstate_dim, agentQuantity, order='F'), axis = 1)
        p_consen_temp = np.kron(1e2*A_Mat[i_agent,:], np.ones([1,ADCstate_dim])).transpose() \
                        * (np.kron(np.ones([agentQuantity,1]), i_agent_Pstate.reshape((-1, 1))) - p.reshape((-1, 1)))
        i_agent_p_consen = np.sum(p_consen_temp.reshape(ADCstate_dim, agentQuantity, order='F'), axis = 1)
        dpdt[slicing] =  i_agent_r_consen - i_agent_p_consen
    return dpdt

def generate_edges(num_nodes):
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    return edges