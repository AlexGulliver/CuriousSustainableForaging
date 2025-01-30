import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

class Node:
    def __init__(self, resource_availability: float, replenishment_rate: float, agent_density: float, behavioural_state: str):
        """
        Node class denoting a particular resource state of the environment. 
        Encodes information about the state of the environment and connects
        to other nodes by edges containing a transition probability.

        """
        self.resource_availability = resource_availability
        self.replenishment_rate = replenishment_rate
        self.agent_density = agent_density
        self.behavioural_state = behavioural_state # Denotes whether there predominanently greedy or moderate agents
        self.transitions = {}

    def add_transition(self, to_node: "Node", probability: float) -> None:
        self.transitions[to_node] = probability

def calculate_transition_probability(from_node: Node, to_node: Node) -> float:
    """
    Calculates transition probability between two nodes based on their attributes.
    """
    if to_node.resource_availability < from_node.resource_availability:
        # Higher probability of depleting resources if agents are greedy
        return 0.7 if from_node.behavioural_state == "greedy" else 0.3
    elif to_node.resource_availability > from_node.resource_availability:
        # Higher probability of replenishment if agents are not greedy
        return 0.5 if from_node.behavioural_state == "not greedy" else 0.2
    else:
        # Neutral transition (e.g., migration)
        return 0.1

# Example: Creating nodes
forest_high = Node(resource_availability=1.0, replenishment_rate=0.8, agent_density=0.2, behavioural_state="not greedy")
forest_low = Node(resource_availability=0.2, replenishment_rate=0.8, agent_density=0.4, behavioural_state="greedy")
desert_high = Node(resource_availability=0.9, replenishment_rate=0.1, agent_density=0.1, behavioural_state="not greedy")

# Add nodes to the graph
G.add_node(forest_high)
G.add_node(forest_low)
G.add_node(desert_high)

# Define transitions between nodes
forest_high.add_transition(forest_low, probability=0.6)  # High resource -> Low resource (e.g., due to over-foraging)
forest_high.add_transition(desert_high, probability=0.4)  # High resource -> Desert high resource (e.g., migration)

# Add edges to the networkx graph with weights
for from_node in G.nodes:
    for to_node in G.nodes:
        if from_node != to_node:
            probability = calculate_transition_probability(from_node, to_node)
            if probability > 0:  # Only add transitions with non-zero probabilities
                G.add_edge(from_node, to_node, weight=probability)



# Assign labels for visualization
labels = {node: f"Resource Availability: {node.resource_availability} - Behavioural State: {node.behavioural_state}" for node in G.nodes}
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

# Draw the graph
pos = nx.spring_layout(G)  # or any layout you prefer
nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()



