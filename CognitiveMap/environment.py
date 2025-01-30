# Math
import numpy as np
import random

# Data handling
from collections import deque
import copy

# Neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Plotting, progress reporting, and file handling
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from agent import BaseAgent, NE_Agent, Greedy_Agent, Moderate_Agent

# Define environment
class Environment:
    def __init__(self, initial_resources, regeneration_rate, carrying_capacity, resource_cap):
        self.resources = initial_resources
        self.regeneration_rate = regeneration_rate
        self.carrying_capacity = carrying_capacity
        self.resource_cap = resource_cap


def simulate_foraging(params_dict):
    """ Core function to simulate foraging agents for a specified set of parameters """

    # unpack required parameters
    env_params, agent_params, pop_params, DQN_params, NE_params, LSTM_params, expmt_params = tuple(params_dict.values())
    expmt_name, num_timesteps, num_runs = tuple(expmt_params.values())

    # initialise environment
    r1, alpha, K, size = tuple(env_params.values())
    num_agents = sum(tuple(pop_params.values()))
    C = r1*num_agents # environment resource cap
    env = Environment(initial_resources=(r1*num_agents), regeneration_rate=alpha, carrying_capacity=K, resource_cap=C)

    # initialise agents
    n_moderate, n_greedy, n_DRQN, n_LSTMDRQN, n_NE, n_LSTMNE = tuple(pop_params.values()) # unpack populations
    agents = [] # create list for agent instances
    
    # initialise required number of agents for each class and add to list
    agents.extend([Moderate_Agent(agent_params) for _ in range(n_moderate)])
    agents.extend([Greedy_Agent(agent_params) for _ in range(n_greedy)])
    agents.extend([DRQN_Agent(agent_params, DQN_params) for _ in range(n_DRQN)])
    agents.extend([LSTM_DRQN_Agent(agent_params, DQN_params, LSTM_params) for _ in range(n_LSTMDRQN)])
    agents.extend([NE_Agent(agent_params, NE_params) for _ in range(n_NE)])
    agents.extend([LSTM_NE_Agent(agent_params, NE_params, LSTM_params) for _ in range(n_LSTMNE)])

    # Create initial observation
    taus = agent_params.get('agent threshold energy')
    n_choices = len(taus)
    observation = [0] * (n_choices + 3)
    count_alive = sum(1 for agent in agents if agent.alive)
    observation[0] = env.resources
    observation[1] = count_alive
    observations = [observation]

    # Main Simulation Loop
    for timestep in range(num_timesteps):
        observation = observations[timestep] # specify current observation
        next_observation = [0] * (n_choices + 3)
        alive_agents = [agent for agent in agents if agent.alive] # create list of currently alive agents

        # for each agent decide action
        for agent in alive_agents:
            agent.decide_action(observation)

        # determine avalible resources based on number of gathering agents
        count_gathering = sum(1 for agent in agents if (agent.gathering == True))
        if count_gathering == 0:
            available_resources = env.resources # handle case where no agents are gathering
        else: available_resources = env.resources/count_gathering # maximum avalible resource for agents
        
        # perform actions for all alive agents
        for agent in alive_agents:
            agent.perform_action(available_resources)
            env.resources -= agent.gather_amounts[-1]
            agent.apply_survival_cost()
            agent.calculate_reward()


        # update resources
        env.resources = env.resources * env.regeneration_rate   # exponential growth of resources
        env.resources = min(env.resources, env.resource_cap)  # cap of max resources

        # construct next observation
        count_alive = sum(1 for agent in agents if agent.alive)
        
        next_observation[0] = env.resources
        next_observation[1] = count_alive
        next_observation[2] = count_gathering
        
        for i in range(n_choices):
            next_observation[3+i] = sum(1 for agent in agents if agent.actions[-1] == i)

        observations.append(next_observation)

        # update networks for agents that are still alive
        alive_agents = [agent for agent in agents if agent.alive] # redetermine alive agents
        for agent in alive_agents:
            if hasattr(agent, 'conduct_learning'):
                agent.conduct_learning(observation, next_observation) # Q-Learning
            if hasattr(agent, 'evolve_networks'):
                agent.evolve_networks() # online neruo-evolution

        # Simulation break conditions
        # early stopping for condition where all agents have died
        if sum(1 for agent in agents if agent.alive) == 0:
            # print(f'All agents are dead at timestep {timestep}')
            break
        
        # early stopping for condition wher all resources are depleted
        # if env.resources == 0:
            # print(f'All resources are depleted at timestep {timestep}')
            break

    # Calaculate statistics
    lifetimes = []
    gather_amounts = []
    total_rewards = []
    for agent in agents:
        lifetimes.append(agent.time_alive)
        gather_amounts.append(sum(agent.gather_amounts))
        total_rewards.append(agent.total_reward)
    mean_lifetime = np.mean(lifetimes)
    mean_gather_amount = np.mean(gather_amounts)
    mean_reward = np.mean(total_rewards)
    
    count_alive = sum(1 for agent in agents if agent.alive)
    survival_rate = count_alive/num_agents

    stats = [mean_lifetime, mean_gather_amount, mean_reward, survival_rate]

    obervations = np.array(observations)
    return obervations, stats

def repeated_simulations(params_dict):
    """ conducts repeated simulations for a given set of parameters for a specified number of runs """
    expmt_name, num_timesteps, num_runs = tuple(params_dict['expmt_params'].values()) # unpack experiment parameters
    run_progbar = tqdm(total=num_runs, desc='Conducting simulations') # create progress bar

    taus = params_dict['agent_params']['agent threshold energy']
    n_cols = len(taus) + 3
    output_shape = (num_timesteps+1, n_cols) # specify expected output shape of observation history

    observations_list = [] # initialise list for storing observation history of each run
    stats_list = [] # initialise list to store run statistics
    
    for _ in range(num_runs):
        observations, stats = simulate_foraging(params_dict) # conduct simulation
        stats_list.append(stats) # record statistics
        
        # check observation history is correct shape and pad if required
        if observations.shape != output_shape:
            n_rows_to_add = output_shape[0] - observations.shape[0]
            observations = np.pad(observations, ((0, n_rows_to_add), (0, 0)), mode='constant')

        observations_list.append(observations) # record observation history
        run_progbar.update(1)

    # calculate overall mean of observation history
    overall_mean_observations = np.mean(observations_list, axis=0)

    # calculate alive agent filtered mean of observation history
    masked_array = np.ma.array(observations_list)
    mask = masked_array[:, :, 1] <= 0
    masked_array = np.ma.masked_array(masked_array, mask=np.repeat(mask[:, :, np.newaxis], n_cols, axis=2))
    filtered_mean_observations = masked_array.mean(axis=0)
    mean_observations = filtered_mean_observations.copy()
    mean_observations[:, 1] = overall_mean_observations[:, 1]

    # calculate mean of run statistics
    mean_stats = np.mean(stats_list, axis=0)
    std_dev_stats = np.std(stats_list, axis=0, ddof=1)

    run_progbar.close()
    return mean_observations, mean_stats, std_dev_stats

def conduct_experiment(params_dict, param_group, param, param_values):
    """ conducts simulations for a range of provided parameters and records run-averaged statistics """
    experiment_params_dict = copy.deepcopy(params_dict) # create copy of params for experiment
    experiment_stats = [] # initialise list to store experiment statistics
    error = []
    
    for i, value in enumerate(param_values):
        experiment_params_dict[param_group][param] = value # modify parameter value in dictionary
        mean_observations, mean_stats, std_dev_stats = repeated_simulations(experiment_params_dict) # run simulation with parameters
        experiment_stats.append(mean_stats) # record run-averaged statistics
        error.append(std_dev_stats[2])

    experiment_stats = np.stack(experiment_stats)

    return experiment_stats, error

def compare_models(params_dict, num_agents=1, keys=None):
    """ Obtains run-averaged observations and stats for all models """
    base_dict = copy.deepcopy(params_dict) # make deep copy of parameter dictionary
    observations, stats = [], []

    if keys == None:
        keys = base_dict['pop_params'].keys()

    for key in base_dict['pop_params'].keys():
        base_dict['pop_params'][key] = 0 # initially set population of all agent types to zero

    for key in keys:
        run_dict = copy.deepcopy(base_dict)
        run_dict['pop_params'][key] = num_agents
        model_observations, model_stats, std_dev_stats = repeated_simulations(run_dict)
        observations.append(model_observations)
        stats.append(model_stats)
    
    stats = np.stack(stats)
    return observations, stats

def plot_observations(observations, title=None, thresholds=None):
    """ Plots forraging simulation results from observation records"""
    plt.rcdefaults()
    plt.style.use('tableau-colorblind10')
    colors = ['#006BA4', '#FF800E', '#A2C8EC', '#FFBC79']
    x_axis = np.arange(len(observations)) # x-axis is number of timesteps
    # agent plotting
    fig, ax1 = plt.subplots(figsize=(9,6))

    for i in range(len(observations[0])-3):
        if thresholds is not None:
            ax1.plot(x_axis, observations[:, (3+i)], '-', color=colors[i], label=r'$\tau$'+f'={thresholds[i]}')
            # ax1.plot(x_axis, observations[:, (3+i)], '-', label=r'$\tau$'+f'={thresholds[i]}')
            # ax1.plot(x_axis, observations[:, (3+i)], '-', label=f'{thresholds[i]}')
        else:
            ax1.plot(x_axis, observations[:, (3+i)], '-', label=f'T{i} agents')
    ax1.plot(x_axis, observations[:, 1], 'k-', label='alive agents')
    ax1.set_xlabel('Time steps', fontsize=18)
    ax1.set_ylabel('Agent count', color='black', fontsize=16)
    ax1.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=18, labelcolor='black')
    # resource plotting
    ax2 = ax1.twinx()
    ax2.plot(x_axis, observations[:, 0], 'k--', label='resources')
    ax2.set_ylabel('Resource in the environment', color='black', fontsize=16)
    ax2.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax2.tick_params(axis='x', labelsize=18, labelcolor='black')

    # add title, labels, and legend
    if title is not None:
        plt.title(title, fontsize=20)
    else:
        plt.title('foraging results', fontsize=16)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filepath = f'plots/{timestamp}.png'
    # plt.savefig(filepath)
    plt.show() # show plot

def plot_results(param, param_values, experiment_stats):
    """ Plots forraging simulation results from observation records"""
    x_axis = param_values
    # agent plotting
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(x_axis, experiment_stats[:, 2], 'g-', label='total reward')
    ax1.set_xlabel('parameter value')
    ax1.set_ylabel('value', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    # resource plotting
    ax2 = ax1.twinx()
    ax2.plot(x_axis, experiment_stats[:, 3], 'k-', label='survival rate')
    ax2.set_ylabel('survival rate', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # add title, labels, and legend
    plt.title('foraging results')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.1, 1))

    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filepath = f'plots/{timestamp}.png'
    # plt.savefig(filepath)
    plt.show()  # show plot

def plot_models(params_dict, observations, agent_types=None, title=None):
    """ Plots forraging simulation results from observation records"""
    plt.style.use('tableau-colorblind10')
    x_axis = np.arange(len(observations[0])) # x-axis is number of timesteps
    
    if agent_types == None:
        agent_types = list(params_dict['pop_params'].keys())

    # plot reward
    fig, ax1 = plt.subplots(figsize=(9,6))
    for i, agent_type in enumerate(agent_types):
        ax1.plot(x_axis, observations[i][:, 1], linestyle='-', label=f'{agent_type} agents')
    ax1.set_xlabel('Time steps', fontsize=18)
    ax1.set_ylabel('Agent count', color='black', fontsize=16)
    ax1.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=18, labelcolor='black')

    # plot survival rate
    ax2 = ax1.twinx()
    for i, agent_type in enumerate(agent_types):
        ax2.plot(x_axis, observations[i][:, 0], linestyle='--', label=f'{agent_type} resources')
    ax2.set_ylabel('Resource in the environment', color='black', fontsize=16)
    ax2.tick_params(axis='y', labelsize=16, labelcolor='black')

    # add title, labels, and legend
    if title is not None:
        plt.title(title, fontsize=20)
    else:
        plt.title('foraging results', fontsize=16)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12)

    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filepath = f'plots/{timestamp}.png'
    # plt.savefig(filepath)
    plt.show() # show plot

def plot_models_without_resource(params_dict, observations, agent_types=None, title=None):
    """ Plots forraging simulation results from observation records"""
    plt.style.use('tableau-colorblind10')
    colors = ['#A2C8EC', '#006BA4', '#FFBC79', '#FF800E', '#bdbdbd', '#000000'] # custom colour mapping
    x_axis = np.arange(len(observations[0])) # x-axis is number of timesteps
    
    if agent_types == None:
        agent_types = list(params_dict['pop_params'].keys())

    # plot reward
    fig, ax1 = plt.subplots(figsize=(9,6))
    for i, agent_type in enumerate(agent_types):
        ax1.plot(x_axis, observations[i][:, 1], linestyle='-', color=colors[i], label=f'{agent_type} agents')
    ax1.set_xlabel('Time steps', fontsize=18)
    ax1.set_ylabel('Agent count', color='black', fontsize=18)
    ax1.tick_params(axis='y', labelsize=18, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=18, labelcolor='black')

    # plot survival rate
    # ax2 = ax1.twinx()
    # for i, agent_type in enumerate(agent_types):
    #     ax2.plot(x_axis, observations[i][:, 0], linestyle='--', label=f'{agent_type} resources')
    # ax2.set_ylabel('Resource in the environment', color='black', fontsize=14)
    # ax2.tick_params(axis='y', labelsize=14, labelcolor='black')

    # add title, labels, and legend
    if title is not None:
        plt.title(title, fontsize=20)
    else:
        plt.title('foraging results', fontsize=16)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, fontsize=12)

    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filepath = f'plots/{timestamp}.png'
    # plt.savefig(filepath)

    plt.show() # show plot

params_dict = {
    'env_params': {
        'initial resources': 500,
        'resource growth rate': 1.005,
        'environment carry capacity': 2500,
        'environment size': 8,
    },
    'agent_params': {
        'agent initial energy': 100,
        'agent carry capacity': 5,
        'agent survival cost': 2,
        'agent threshold energy': [30, 50, 80, 50000], # threshold choices available to agents
        'reward scaling': 100
    },
    'pop_params': { # dictates number of agents of each type to be simulated
        'moderate': 0,
        'greedy': 0,
        'DRQN': 0,
        'LSTM-DRQN': 0,
        'NE': 0,
        'LSTM-NE': 0
    },
    'DQN_params': { # Q-learning parameters used for DRQN agents
        'learning rate': 0.025,
        'discount factor': 0.95,
        'exploration rate': 1,
        'exploration decay': 0.99
    },
    'NE_params': { # EA parameters used for online neuroevolution agents
        'pop_size': 30,
        'tournament size': 5,
        'mutation rate': 0.2,
        'mutation scale': 0.06
    },
    'LSTM_params': { # parameters for LSTM networks
        'observation length': 25, # length of series of observations to be passed to LSTM
        'reward length': 1, # not used in current implementation
        'discount factor': 0.95
    },
    'expmt_params': {
        'experiment name': 'None',
        'num timesteps': 1000,
        'num runs': 30
    }
}

agent_types = ['moderate', 'greedy'] # specify models for comparison, if none all models are compared
observations, stats = compare_models(params_dict, num_agents=1, keys=agent_types) # run model comparison
print(stats) # [mean lifetime, mean amount gathered, mean total reward, survival rate]
title = 'Moderate and greedy baseline agents'
plot_models(params_dict, observations, agent_types, title) # Plot model comparison