# Math
import numpy as np
import random

# Define base agent class: used by all agents manages properties and actions that are the same for all agents

class BaseAgent:
    def __init__(self, agent_params):
        """ Initializes base agent with provided parameters """
        eta, self.carry_capacity, self.survival_cost, self.tau, self.k = tuple(agent_params.values())
        self.energy = random.randint(eta-10, eta+10) # fuzzy eta avoids all agents gathering on same time step

        self.alive = True
        self.time_alive = 0

        self.actions = []
        self.gathering = False
        self.gather_amounts = []
        self.current_reward = 0
        self.total_reward = 0
        self.all_rewards = []
    
    def perform_action(self, available_resources):
        """ Determines how many resources are gathered by agents and adds this to their energy """

        if self.gathering == True:
            #TODO : Add validation check for agent position as compared to resource
            self.gather_amounts.append(min(available_resources, self.carry_capacity))
        else:
            self.gather_amounts.append(0)

        self.energy += self.gather_amounts[-1] # add gathered resources to energy
        
    def apply_survival_cost(self):
        """ Subtracts survival cost from agent energy and sets agent to dead if no energy remaining """
        
        self.energy -= self.survival_cost # reduce agent energy by survival cost
        if self.energy <= 0:
            self.alive = False # if agent is out of energy then agent dies
        else:
            self.time_alive += 1 # if agent is still alive increment time alive counter

    def calculate_reward(self):
        """ Calculates logarithmic reward based on agent energy """
        if self.energy <= 0:
            reward = 0 # if agent has no energy reward is zero
        else:
            reward = self.k * np.log(self.energy) # agent reward based on log of energy level
        
        # update rewards
        self.current_reward = reward
        self.total_reward += reward
        self.all_rewards.append(reward)

# Define base classical agent classes
# Strategy is predetermined and fixed
# Base moderate agent only takes moderate actions, or the most moderate action given a range of actions
# Base greedy agent only takes greedy actions

class Moderate_Agent(BaseAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params)
    
    def decide_action(self, observation):
        action = 0 # always chooses lowest threshold value
        if self.energy <= self.tau[action]:
            self.gathering = True
        else:
            self.gathering = False
                
        self.actions.append(action)

class Greedy_Agent(BaseAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params)
    
    def decide_action(self, observation):
        action = len(self.tau)-1 # always chooses highest threshold value
        if self.energy <= self.tau[action]:
            self.gathering = True
        else:
            self.gathering = False
                
        self.actions.append(action)

# class NE_Agent(BaseAgent):
#     def __init__(self, agent_params, NE_params):
#         super().__init__(agent_params)

#         # load parameters from parameter dictionary
#         self.pop_size, self.tournament_size, self.mutation_rate, self.mutation_scale = tuple(NE_params.values())

#         # initialise networks
#         self.n_choices = len(self.tau)
#         self.network = DeepQNetwork(input_dim=self.n_choices+3, output_dim=self.n_choices)
#         self.networks = [DeepQNetwork(input_dim=self.n_choices+3, output_dim=self.n_choices) for _ in range(self.pop_size)] # create poputlation of networks
#         self.fitness_scores = np.array([(self.k * np.log(self.energy)) for _ in range(self.pop_size)])
#         self.selected_network_index = None

#     def decide_action(self, observation):
#         # optionally use tournamet selection rather than softmax prior to specified timestep
#         if self.time_alive < 0:
#             self.selected_network_index = tournament_selection(self.fitness_scores)
#         else:
#             # determine which network to use based on softmax selection of fitness scores
#             self.selected_network_index = softmax_selection(self.fitness_scores)
#         selected_network = self.networks[self.selected_network_index]
        
#         # use selected network to determine energy threshold for gathering
#         with torch.no_grad():
#             observation_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Convert observation to tensor
#             output = torch.softmax(selected_network(observation_tensor), dim=1)
#             action = np.argmax(output).item()
#         self.actions.append(action)

#         # set gathering to True if energy below threshold or False if energy above threshold
#         if self.energy <= self.tau[action]:
#             self.gathering = True
#         else:
#             self.gathering = False

#     def evolve_networks(self):
#         """ Evolves network population based on network fitness scores """
#         self.fitness_scores[self.selected_network_index] = self.current_reward # update reward for last used network
#         worst_index = np.argmin(self.fitness_scores) # obtain index of worst performing network

#         # specify parent networks
#         idx1 = tournament_selection(self.fitness_scores)
#         idx2 = tournament_selection(self.fitness_scores)
#         parent1 = self.networks[idx1]
#         parent2 = self.networks[idx2]

#         # determine child fitness from average of parent fitnesses
#         parent_avg_fitness = (self.fitness_scores[idx1] + self.fitness_scores[idx2]) / 2

#         child = arithmetic_crossover(self.network, parent1, parent2) # perform crossover of parent networks
#         mutate_network(child) # perform mutation of child network

#         # replace weakest network in population with child network
#         self.networks[worst_index] = child
#         self.fitness_scores[worst_index] = parent_avg_fitness