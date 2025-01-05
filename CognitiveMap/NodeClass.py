import networkx as nx

G = nx.Graph()

class Node:
    def __init__(self, resource_availability, resource_replenishment, agent_density, behavioural_state):
        self.resource_availability = resource_availability
        self.resource_replenishment = resource_replenishment
        self.agent_density = agent_density
        self.behavioural_state = behavioural_state
