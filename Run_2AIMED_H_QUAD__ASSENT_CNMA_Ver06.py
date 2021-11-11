#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:18:58 2020

@author: bizon
"""
# In[Imports]:

import dill

import sys
from sklearn.metrics import mean_squared_error
from joblib import dump, load
# Import other librariesglobal_sobol_counter
import numpy as np
import random
import copy
# import pygmo as pg
from tqdm import tqdm
# import runcommand
# import pandas as pd
import os
import sobol_seq
import itertools
import datetime
import dill
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pygmo as pg
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from smt.sampling_methods import LHS

from diversipy import psa_partition, psa_select, select_greedy_maximin, select_greedy_maxisum, select_greedy_energy
from flight_dynamics import SWRIFlightDynamics

# In[]:
# In[]:

sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/") # used for runcommand ffile - update accordingly
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/nsverify") # Uses for MILP formulation - update accordingly
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/envs/code/env_code/environments/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/envs/nsverify/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICEmyMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/nsverify")
sys.path.append("/della/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits")
sys.path.append('/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/CNMAWithBO')
sys.path.append('sratch/gpfs/pterway/HSPICE/GAComponentSelection/GAComponentSelectionMitArch/GACompSelSetup')
# sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/Marabou")
# sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/NNet")
sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/NNet/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/")
sys.path.append("/della/scratch/gpfs/pterway/HSPICE/HSPICE/ThreeStage/CNMAOnOriginalCircuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/CNMAOnOriginalCircuit/")
sys.path.append("/scratch/gpfs/pterway/perspectaTestsVer2/perspectaV1/myCodesV9Della/")
sys.path.append("/scratch/gpfs/pterway/perspectaProject/myCodesV9Della/")
sys.path.append("/scratch/gpfs/pterway/perspectaTestsVer2/perspectaV1/myCodesV9Della/project2BOTOrchBAsedVer01/dexcelMOOVer02")

# In[Load libraries]:

# import pygmo as pg
# import pandas as pd
import sobol_seq
# In[Imports Continued]: Do imports
# from gurobipy import *
from sklearn.preprocessing import MinMaxScaler

# In[Add paths to system]"
sys.path.append("/tigress/pterway/perspectaTests/perspectaV1/myCodesV2/myCodes") # used for runcommand ffile - update accordingly
sys.path.append("/tigress/pterway/perspectaTests/perspectaV1/myCodesV2/myCodes/nsverify") # Uses for MILP formulation - update accordingly
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/") # used for runcommand ffile - update accordingly
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/nsverify") # Uses for MILP formulation - update accordingly
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/envs/code/env_code/environments/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/envs/nsverify/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICEmyMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/gym_circuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master/gym_circuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/myMitCircuits/Tic-Tac-Toe-Gym_Environment-master")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/nsverify")
sys.path.append("/della/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits")
sys.path.append('/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/CNMAWithBO')
sys.path.append('sratch/gpfs/pterway/HSPICE/GAComponentSelection/GAComponentSelectionMitArch/GACompSelSetup')
# sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/Marabou")
sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/NNet")
sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/NNet/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/ThreeStage/MarabouInvDesign/")
sys.path.append("/della/scratch/gpfs/pterway/HSPICE/HSPICE/ThreeStage/CNMAOnOriginalCircuit/")
sys.path.append("/scratch/gpfs/pterway/HSPICE/CNMAOnOriginalCircuit/")
sys.path.append('/scratch/gpfs/pterway/perspectaTestsVer2/perspectaV1/myCodesV9Della/MarabouPyJune16th2021/Marabou')


# In[Perspecta]:
import types

import numpy as np
import copy
import os
import sobol_seq
import dill
# In[Imports Continued]: Do imports
from gurobipy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dense
from generateSobolSamplesAroundNominal import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
# from getMissDataMissingV1.py import *
import time
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import multiprocessing
# In[Perspecta]:
import types
# from keras.callbacks import ModelCheckpoint
# import keras
# from keras.callbacks import ModelCheckpoint
# import keras
# device = torch.device("cpu")
# 
global_simulations_done = 0
# In[]:
keep_data_len = 1000

# To run set up a input dictionary - parameter names are in config.dexcel

# In[]:
nnet_append = np.random.randint(100)
print('Marabou will have this at end ', nnet_append)
# In[]:
# Check sinc plot
# rng_snc = np.arange(start = 0, stop = 100, step = 1)
# clip_value = np.zeros_like(rng_snc) + 0.1
# # sincx = np.minimum(clip_value, np.abs(np.sinc(0.5*rng_snc)) + 0.01)
# sincx = np.minimum(clip_value, np.abs(np.sinc(0.1*rng_snc))/5 + 0.01)

# plt.plot(rng_snc,sincx )
# In[]:
t = time.time()
np.random.seed(10)
parallel_simulation_time_tracked = []
enc = 1
# In[]:
params = types.SimpleNamespace()
# params.dt = 0.2
actual_sim_counter = 0
sobols_used = 0
total_simulation_till_now = 0
arch_to_try = [(100), (15), (40,20,8), (20,20,8)] 
# arch_to_try = [(30), (15), (40,20,8)] 

# In[]:
# My code
# These are the nominal values around which samples are generated

# Order is:  control_Q_position, control_Q_velocity, control_Q_angular_velocity, 
# control_Q_angles, control_R, control_requested_lateral_speed, 
# control_requested_vertical_speed

lower_bound = [0, 0, 0, 0, 0, 0 , -5, ]
upper_bound = [5, 5, 5, 5, 5, 40, 0]


num_cores = multiprocessing.cpu_count()
print('Number of cores are', num_cores)
# Create sobol sequence
max_samples = 10000
arr_list = []
for j in range(0,len(lower_bound)):
  lb = lower_bound[j]
  ub = upper_bound[j]
  this_ele = [lb, ub]
  arr_list.append(this_ele)
  

xlimits = np.array(arr_list)
sampling = LHS(xlimits = xlimits, random_state = 10)
raw_samples = sampling(max_samples)
# seed1 = [0.05607722723439856,2.1587003455109315,31969274557303978, 1, 14.841766506830323,20.047591231707173,0 ]

# seed1_array = np.array(seed1).reshape(1,-1)

# raw_samples[0]  =seed1_array
# raw_samples[0] =  [0.05607722723439856,2.1587003455109315,31969274557303978, 1, 14.841766506830323,20.047591231707173,0 ]
# =# sobol_samples = sobol_seq.i4_sobol_generate(len(lower_bound_small), max_samples)
# diff_max_min = (np.array(upper_bound_small)
#               - np.array(lower_bound_small)).reshape(1,-1)
# diff_repeated = np.tile(diff_max_min, (sobol_samples.shape[0], 1))

# raw_samples = ( np.array(lower_bound_small).reshape(1,-1) +
#                 (np.multiply(diff_repeated, sobol_samples))
#                 )
# In[Store experiences]:
stored_inputs = []
stored_objectives = []
stored_simulated_outputs  = []
stored_simulated_outputs_with_success = []
complete_step1_step2_time_limit = 3600*10
time_out = 3600*7 # Step 1 budget
# time_out = 30*2 # Step 1 budget

switch_enable = False
global_sobol_counter = 0
step2_enabled = False
# In[]:
# In[]:


# In[]:
# Create an individual
def create_individual(current_try=0):
    global global_sobol_counter
#    available_components = list(parts.keys())
    print('Sampled Sobol individual is ')
    sobol_sample = raw_samples[global_sobol_counter,:]
    print(sobol_sample)
    global_sobol_counter+=1
    print('Total sobol samples are', global_sobol_counter)
    return sobol_sample.tolist()
# In[Population creation]:
# Define the function to create a population
def create_population(individuals = 10):
    """
    Create random population with given number of individuals
    Input: Number of individuals required
    Output: All members of the population

    """
    population = []
    # Loop through each row (individual)
    for i in range(individuals):
        this_individual = create_individual(i)
        population.append(this_individual)
    return population
# In[check initial population]:
# init_pop = create_population(individuals = 10)
# In[]:
# In[]:
# In[Train keras model]:
def trainSklearnmodel(normalized_inputs, normalized_outputs, arch_to_use):
  X_train, X_test = normalized_inputs, normalized_outputs

  clf = MLPRegressor(hidden_layer_sizes=arch_to_use,
                      solver='adam', verbose=0, tol=1e-20,
                      learning_rate_init=.0001,  learning_rate='adaptive', max_iter=100000)       
  # clf = MLPRegressor(hidden_layer_sizes=arch_to_use,
  #                     solver='adam', verbose=0, random_state=1, tol=1e-20,
  #                     learning_rate_init=.0001,  learning_rate='adaptive', max_iter=100000)    
  clf.fit(X_train, X_test)  
  loss = clf.loss_
  return loss, clf, arch_to_use

# In[dic from ind]:
best_mse, best_model, best_arch = None, None, None
nn_update_frequency = 50
def getBestNNArchSklearn(normalized_inputs, normalized_outputs, arch_choices):
  global best_arch, step_2_calls, best_mse, best_model, best_arch
  if (step_2_calls%nn_update_frequency == 0) or (best_mse == None):
    print('Getting the best NN architecture')
    mse_track = []
    clf_track = []
    arch_track =[]
    cores_to_use_parallel = min(len(arch_choices), num_cores)
    tracked_loss_clf = Parallel(n_jobs=cores_to_use_parallel)(delayed(trainSklearnmodel)(normalized_inputs, normalized_outputs, ar) for ar in arch_choices)
    for res in tracked_loss_clf:
      this_mse, this_clf, this_arch = res
      mse_track.append(this_mse)
      clf_track.append(this_clf)
      arch_track.append(this_arch)
    
    mse_track_array = np.array(mse_track)
    best_model = clf_track[np.argmin(mse_track_array)]
    best_arch = arch_track[np.argmin(mse_track_array)]
    best_mse = np.min(mse_track_array)
    print('All MSEs are', mse_track_array)
    print('Best MSE is ', np.min(mse_track_array))
    print('The Chosen architecture is ', best_arch)
    
    print('Retrained MSE is ', best_mse)
  return best_mse, best_model, best_arch 

# In[dic from ind]:

    
# In[]:
# Computer individual fitness

# simulator = SWRIFlightDynamics(**{'template_file': 'america-quad.txt'})
# simulator = SWRIFlightDynamics(**{'template_file': 'path_5_R2.inp'})
simulator = SWRIFlightDynamics(**{'template_file': 'FlightDyn_quadH.inp'})

# simulator = SWRIFlightDynamics(**{'template_file': 'FlightDyn_7By3.inp'})
# simulator = SWRIFlightDynamics(**{'template_file': 'Quad1PathV1.inp'})
# simulator = SWRIFlightDynamics(**{'template_file': 'path_5_R2.inp'})
# In[]:
def computeFitnesIndividual(ind):
    global total_simulation_till_now
    # total_simulation_till_now+=1
    # print('Simulations done till now is ', total_simulation_till_now)
    elapsed = time.time() - t 
    print('Time elapsed till now is ', elapsed)
    
    print('This individual is',ind)
    print('--'*50)
    if type(ind) == tuple:
      this_ind_tuple = ind
      ind = list(this_ind_tuple)
    if ind in stored_inputs:
        index_location_input = stored_inputs.index(ind)
        print('Using Memoization')
        print('fuel_objective, time_objective, reward_objective', stored_objectives[index_location_input])
        print('Simulated outputs are from Memoization ', stored_simulated_outputs[index_location_input])
        return stored_objectives[index_location_input]
    print('Performing simulation ')

    # x = dict(battery_capacity=6000.0, # Fixed to 6000 for all cases
    #         control_i_flight_path=5, # Fix to one of these 1,3,4,5
    #         control_Q_position=ind[0],
    #         control_Q_velocity=ind[1],
    #         control_Q_angular_velocity=ind[2],
    #         control_Q_angles=ind[3],
    #         control_R=ind[4],
    #         control_requested_lateral_speed=ind[5],
    #         control_requested_vertical_speed=ind[6],
    #         )   
    x = dict(
            control_i_flight_path=5, # Fix to one of these 1,3,4,5
            control_Q_position=ind[0],
            control_Q_velocity=ind[1],
            control_iaileron = 5,
            control_iflap = 6,
            control_Q_angular_velocity=ind[2],
            control_Q_angles=ind[3],
            control_R=ind[4],
            control_requested_lateral_speed= ind[5],
            control_requested_vertical_speed=ind[6],
            )     
    # x = dict(
    #         control_i_flight_path=5, # Fix to one of these 1,3,4,5
    #         control_Q_position=ind[0],
    #         control_Q_velocity=ind[1],
    #         control_Q_angular_velocity=ind[2],
    #         control_Q_angles=ind[3],
    #         control_R=ind[4],
    #         control_requested_lateral_speed=ind[5],
    #         control_requested_vertical_speed=ind[6],
    #         ) 
    
    this_simulation = simulator.sim(x)
    this_score = this_simulation['Path_traverse_score_based_on_requirements']
    # this_simulation = ff.run_f('lunar_lander', ind, params)
    # this_ind_obj = ff_obj.objectives
    # this_ind_constraint = ff_obj.constraints
    total_simulation_till_now+=1  
    print('Simulated output is ', this_score)
    print('Total simulations done till now is ', total_simulation_till_now)
    # this_simulation = [this_simulation[0], this_simulation[1], this_simulation[3]]
    # fuel_used, time_taken, reward_obtained = this_simulation
    obj1 = copy.copy(this_score)

    # fuel_used, time_taken, reward_obtained = this_simulation
    # if not((float("inf")  in this_simulation) or (-float("inf")  in this_simulation)):
    stored_simulated_outputs.append([ obj1])
    # if (float("inf")  in this_simulation) or (-float("inf")  in this_simulation):
    stored_inputs.append(ind)
    print(' obj1', this_score)

    multi_objective = [-obj1]
    print('obj1', multi_objective)
    # stored_inputs.append(ind)
    stored_objectives.append(multi_objective)
    # stored_simulated_outputs.append([fuel_used, time_taken, reward_obtained ])
    return multi_objective
# In[]:
# fit_test = computeFitnesIndividual(init_pop[3])
# print('fit_test', fit_test)
# test_ind = [0.8, 5, 4, 200, 1.33, 360, 0.6, 20, 0.0078, 0.1, 0.55, 1025, 0.4]
# test_ind = [0.9067179040881624, 5.0, 4.0, 200.0, 1.33, 360.0, 0.6, 20.0, 0.0078, 0.1, 0.55, 1030.0, 1.5]
# fit_test = computeFitnesIndividual(test_ind)
# print(fit_test)

# In[check fitness]:
# fit_test = computeFitnesIndividual(init_pop[4])
# print('fit_test', fit_test)
# In[One Point Crossover]:
def cxOnePoint(individual1, individual2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    ind1, ind2 = list(individual1), list(individual2)

    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    new_ind1 = (ind1)
    new_ind2 = (ind2)

    return new_ind1, new_ind2
# In[Test One Point Cross]:

# c1,c2 = cxOnePoint(init_pop[0], init_pop[1])

# In[Tournament selection]:

def selTournamentNSGA(individuals, k, tournsize, fitness):

    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fitness: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """

    # Generate a population if size N. Selection is based on dominated
    # front and not fitness
    chosen = []
    population_size = len(individuals)
    # print((fitness))
    # print(type(fitness))
    # sorted_population = pg.sort_population_mo(points = fitness) # Disable for single objective - use numpy
    # print(sorted_population)
    # print(type(sorted_population))
    sorted_population = np.argsort(np.array(fitness).reshape(-1,)).reshape(-1,)
    # print(sorted_population)
    # print(type(sorted_population))
    population_rank = np.zeros((population_size,))
    for j in range(0,population_size):
        population_rank[j] = np.where(sorted_population==j)[0]

    chosen = []
    selected_members = []
    for i in range(k):
        selected_individuals = np.random.choice(population_size,
            tournsize, replace=False)
        # selected_individuals_list = selected_individuals.tolist()
        these_individual_rank = population_rank[selected_individuals]
        sorted_rank = np.argsort(selected_individuals)
        best_individual = int(these_individual_rank[sorted_rank[0]])
        chosen.append(best_individual)
        selected_members.append(individuals[best_individual])
    return chosen, selected_members

# In[Mutation]:
# In[]:
# NOTE: The population generated through this will need to be adjusted based on the constraints specified
# Such as the first node should be capacitor, second should be .....
# Simplify - allow the circuit to have one and only one bjt!!!
def mutation(chromosome, mutation_rate = 0.4):
    '''
    mutation acts only on the main string. It selects another component ,
    its end terminals and its value
    Perform mutation, but dealso ensure that for this chromosome too,
    we are connected to the ground terminal for gene1
    and output terminal for the second gene. Similar to what was done
    earlier
    Activation mask remains unchanged
    '''
    global global_sobol_counter, raw_samples
    genes = list(chromosome)
    # perform mutation at each locus of the chromosome with mutation
    # probability of 2%, but make sure the first and second gene
    # are active and satisfy terminal conditions
    new_gene = []

    # HARD Coded - Need to Change!!!
    # Generate predetermined samples - ie. just create individual
    # Pass the individual number
    ind_number = random.sample(np.array(np.arange(global_sobol_counter)).tolist(),1)[0]
    new_chromosome = raw_samples[ind_number]

    for i,gene in enumerate(genes):
        if random.uniform(0, 1) < mutation_rate:
            # mutate this gene
            this_gene = new_chromosome[i]

        else:
            this_gene = gene
        new_gene = new_gene + [this_gene]
    return tuple(new_gene)


# In[]:
def generateNewPopulationNSGA(oldPopulation, population_fitness ,number_elites = 2,
    tournament_size = 3, crossover_prob = 0.7, mutationn_prob = 0.02):
    '''
    based on the fitness of members in the population, select the elites and pass
    them to the next generation. To generate other members of the population, select
    two individuals with probability proportional to the fitness.
    Perform mutation, crossover and sweep to generate new members
    https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
    '''
    # Sort the population based on the fitness
    # Generate new individuals
    number_of_individuals_required = len(oldPopulation)
    chosen_list, mating_population = selTournamentNSGA(oldPopulation,
    number_of_individuals_required, tournament_size, population_fitness)
    # Replace this with NSGA selection
    # TO DO
    # Do crossover and mutation on these individuals denoted by
    # mating_population
    mated_population = []
    for j in range(0,number_of_individuals_required):
        # pick two individuals randomly
        parents = np.random.choice(number_of_individuals_required, 2, replace = False)
        mother = mating_population[parents[0]]
        father = mating_population[parents[1]]
        # decide if we need to do crossover
        if random.uniform(0, 1) < crossover_prob:
            child1, child2 = cxOnePoint(mother, father)
        else:
            child1, child2 = mother, father

        # decide whether to perform mutation on this
        child1_chromosome = child1
        child2_chromosome = child2


        child1_chromosome_mutated = mutation(child1_chromosome)
        child2_chromosome_mutated = mutation(child2_chromosome)

        child1 = child1_chromosome_mutated
        child2 = child2_chromosome_mutated
        # make sure terminals are correct

        mated_population = mated_population + [child1]
        if len(mated_population) >= (number_of_individuals_required):
            break
        mated_population = mated_population + [child2]
        if len(mated_population) >= (number_of_individuals_required):
            break


    # Correct nodes of mated population: not required here as we are only selecting components
    mated_population_nodes_corrected = mated_population
    # Now compute fitness of the mated population
    no_objectives = population_fitness.shape[1]
    fitness_array_mated = np.zeros((len(mated_population_nodes_corrected),
     no_objectives))

    for i,pop in tqdm(enumerate(mated_population_nodes_corrected)):

        this_fitness = computeFitnesIndividual(pop)
        if (float("inf")  in this_fitness) or (-float("inf")  in this_fitness):
           print('Failed candidate is' )
           print(mated_population_nodes_corrected[i])
           # start generating Sobol samples and replace the design, store the simulation too
           failed_simulation = True
           while (failed_simulation):
               # create an indiivdual
               print('Doing simulation until finite output')
               print('**'*50)
               new_candi = create_individual()
               this_fitness = computeFitnesIndividual(new_candi)
               failed_simulation = (float("inf")  in this_fitness) or (-float("inf")  in this_fitness)
           # replace the individual
           mated_population_nodes_corrected[i] = new_candi
           print('Replaced candidate is ')
           print(mated_population_nodes_corrected[i])         
        fitness_array_mated[i,:] = np.array(this_fitness)
    combined_population = oldPopulation + mated_population_nodes_corrected
    combined_fitness = np.concatenate((np.array(population_fitness),
        fitness_array_mated), axis = 0)
    number_of_individuals_required = len(oldPopulation)
    # Disable for single objective
    # best_individuals = pg.select_best_N_mo(points = combined_fitness,
    #     N=number_of_individuals_required)
    best_individuals = np.argsort(np.array(combined_fitness).reshape(-1,)).reshape(-1,)
    best_individuals = best_individuals[:number_of_individuals_required]
    # pg.select_best_N_mo(points = combined_fitness,
    #     N=number_of_individuals_required)
    best_individuals_array = best_individuals.tolist()
    fitness_new_generation = np.zeros((number_of_individuals_required, combined_fitness.shape[1]))
    ind = 0
    new_population = []
    for j in best_individuals_array:
        new_population = new_population + [combined_population[j]]
        fitness_new_generation[ind,:] = combined_fitness[j,:]
        ind = ind + 1


    assert(len(new_population) == len(oldPopulation))
    return new_population,fitness_new_generation

# In[]:    


  
  
# In[Step 2 simulations]:
# def getTopParetoFront(stored_inputs= stored_inputs, stored_simulated_outputs=stored_simulated_outputs, number_of_top_individuals=10):
def getTopParetoFront(number_of_top_individuals=10):  
  global stored_inputs, stored_simulated_outputs
  stored_inputs_array_not_ohe = np.array(stored_inputs)
  # stored_simulated_outputs.append([ obj1, time_taken,  is_success , obj2])
  stored_simulated_outputs_array_all = np.array(stored_simulated_outputs)

  top_outputs_sorted = np.argsort(-np.array(stored_simulated_outputs_array_all).reshape(-1,)).reshape(-1,)
  top_outputs = stored_simulated_outputs_array_all[top_outputs_sorted[:number_of_top_individuals]]
  top_inputs = stored_inputs_array_not_ohe[top_outputs_sorted[:number_of_top_individuals]]

  top_outputs_list = top_outputs.tolist()
  top_inputs_list = top_inputs.tolist()
  return top_inputs_list, top_outputs_list 

# In[]: 
 
# In[Clip inputs]:
def clipInputsWithinRange(input_to_clip):
  global lower_bound, upper_bound
  clipped_input = []
  for bound_iter, comp_in in enumerate(input_to_clip):
    this_clip = comp_in
    if comp_in> upper_bound[bound_iter]:
      this_clip = upper_bound[bound_iter]
    if comp_in< lower_bound[bound_iter]:
      this_clip = lower_bound[bound_iter]
    clipped_input.append(this_clip)
  return clipped_input
      
# In[CNMA candidate solution]:
#######################################################################################################
    
# In[]:
def generateEachCNMACandidateSolution(model,
                                           bo_inputs_lower_bound_raw_scaled,
                                           bo_inputs_upper_bound_raw_scaled,
                                           bo_outputs_lower_bound_raw_scaled,
                                           bo_outputs_upper_bound_raw_scaled, 
                                           enc,
                                           scaler_input_bo,
                                           scaler_output_bo,
                                           des_out):
  # generate candidate solution
  # In[]:
  gmodel = Model("Test") # Gurobi model
  gmodel.Params.MIPGap = 1e-6
  gmodel.Params.FeasibilityTol = 1e-7
  gmodel.Params.IntFeasTol = 1e-6
  gmodel.Params.LogToConsole = 0 
  gmodel.Params.TimeLimit = 50;
  wrapper_gmodel = NetworkModel(gmodel) # See its definition
  dense_mod, relu_mod = wrapper_gmodel.add_vars(model.layers) # Create all the variables based on NN architecture   
  gmodel.update() # Update gurboi model 
  # In[write input variables to the model]:  
  input_gurobi_variables_g = []
  var_number = 0 
  number_of_inputs = len(bo_inputs_lower_bound_raw_scaled)
  for i in range(number_of_inputs):
    input_gurobi_variables_g.append(gmodel.addVar(lb = bo_inputs_lower_bound_raw_scaled[var_number],
                                                      ub = bo_inputs_upper_bound_raw_scaled[var_number], 
                                                      name = 'var_' + str(var_number)))   
    # if (i<number_of_inputs-5):
    #     input_gurobi_variables_g.append(gmodel.addVar(lb = bo_inputs_lower_bound_raw_scaled[var_number],
    #                                                   ub = bo_inputs_upper_bound_raw_scaled[var_number], 
    #                                                   name = 'var_' + str(var_number)))
    # else:
    #     input_gurobi_variables_g.append(gmodel.addVar(lb = bo_inputs_lower_bound_raw_scaled[var_number],
    #                                                   ub = bo_inputs_upper_bound_raw_scaled[var_number], 
    #                                                   name = 'var_' + str(var_number), 
    #                                                   vtype=GRB.BINARY))   
        # print('Binary added')
    var_number +=1
 
  gmodel.update()
  inputs_model = input_gurobi_variables_g   
    
  # In[]:    
  constraints_NN_output = wrapper_gmodel.add_constraints(model.layers,
                                                  inputs_model,
                                                  dense_mod, relu_mod)    
    
    
  # In[]:
  cnma_obj1 = constraints_NN_output[0]   
  # cnma_obj2 = constraints_NN_output[3] 
  # cnma_con1 = constraints_NN_output[1] 
  # cnma_con2 = constraints_NN_output[2] 

  
  # In[]:
  
  # In[]:
  # cnma_obj.lb =   bo_outputs_upper_bound_raw_scaled[0]
  # improvement_required = np.random.uniform(low = 1e-3, high = 1e-2)
  cnma_obj1.lb = des_out[0]
  # cnma_obj2.lb = des_out[3]
  # cnma_con1.ub = bo_outputs_lower_bound_raw_scaled[1]
  # cnma_con2.lb = bo_outputs_lower_bound_raw_scaled[2]

  # cnma_obj1.lb =   des_out[0]
  # cnma_obj2.lb =   des_out[1]
  # cnma_con1.lb = bo_outputs_lower_bound_raw_scaled[2]
  # cnma_con2.lb = bo_outputs_lower_bound_raw_scaled[3]
  # cnma_con3.lb = bo_outputs_lower_bound_raw_scaled[4]
  # cnma_con4.lb = bo_outputs_lower_bound_raw_scaled[5]
  
  # cnma_con1.ub = bo_outputs_upper_bound_raw_scaled[1]
  # cnma_con2.ub = bo_outputs_upper_bound_raw_scaled[2]
  # cnma_con3.ub = bo_outputs_upper_bound_raw_scaled[4]
  # cnma_con4.ub = bo_outputs_upper_bound_raw_scaled[5]  
      
      
      
  # In[]: 
  # gmodel.addConstr(cnma_con2 == 1)
  # In[]:      
  epsilons = quicksum((quicksum(e) for (e, _, _) in dense_mod))
  gmodel.addConstr(epsilons <= 1e-3)
  # Update Gurobi model
  gmodel.update()
  # Optimize
  gmodel.optimize()  

  # In[]: 
  if gmodel.status != GRB.OPTIMAL:
    return [False]
  if gmodel.status == GRB.OPTIMAL:
    elapsed = time.time() - t 
    print('Time elapsed till now is ', elapsed)
    
  # In[]:
    # Get suggested inputs by Gurobi
  suggested_inputs_gb = []
  var_number = 0
  for i in range(number_of_inputs):
      suggested_inputs_gb.append(input_gurobi_variables_g[var_number].x)
      var_number +=1  
        
  predicted_outputs_gb = []
  var_number = 0
  for out_variable in constraints_NN_output:
      predicted_outputs_gb.append(out_variable.x)
      var_number +=1    
   # In[]:
  NN_outputs = np.array(predicted_outputs_gb)
  NN_outputs_raw = scaler_output_bo.inverse_transform(NN_outputs.reshape(1,-1))
  print('Predictions by Neural Network Are: ')
  print( NN_outputs_raw)    
  inp_op_con_satisfied = False
  NN_inputs = np.array(suggested_inputs_gb)
  raw_inputs = scaler_input_bo.inverse_transform(NN_inputs.reshape(1,-1)).reshape(-1,)
  raw_inputs_list = [raw_inputs.tolist()]      
  # In[]:
  estimated_inputs_array_denormalized_list_clipped = []
  for est_in_gmm in raw_inputs_list:
      clipped_input_to_range = clipInputsWithinRange(est_in_gmm)
      
      estimated_inputs_array_denormalized_list_clipped.append(clipped_input_to_range)
  estimated_inputs_array_denormalized_list_clipped_array = np.array(estimated_inputs_array_denormalized_list_clipped)  
  # In[]:
  return estimated_inputs_array_denormalized_list_clipped
   

# In[]:
def generateSingleCNMACandidateSolution(top_inputs ,top_outputs, number_of_req_per_ind = 10):
  # In[]:
  # In[]:
    
  global stored_inputs, stored_simulated_outputs  , gen_since_last_improved , no_improve_count_step2, step2_enabled, keep_data_len
  print('Doing parallel CNMA')
  stored_inputs_array_not_ohe = np.array(stored_inputs)
  
  stored_simulated_outputs_array_all = np.array(stored_simulated_outputs)
  
  unique_gmm_inputs, ubique_gmm_inputs_indices = np.unique(
                        np.array(stored_inputs_array_not_ohe.tolist()), return_index=True, return_inverse=False, return_counts=False, axis=0)
  unique_gmm_outputs = stored_simulated_outputs_array_all[ubique_gmm_inputs_indices,:]  
  
  stored_simulated_outputs_array_all = np.copy(unique_gmm_outputs)
  stored_inputs_array_not_ohe = np.copy(unique_gmm_inputs)
  
  # con_satisfied_indices = np.where((stored_simulated_outputs_array_all[:,2]==1) & 
  #                                  (stored_simulated_outputs_array_all[:,1]<=time_max)                                     
  #                                  )
  # stored_simulated_outputs_array = stored_simulated_outputs_array_all[con_satisfied_indices[0],:]
  # stored_simulated_inputs_con_sat_array = stored_inputs_array_not_ohe[con_satisfied_indices[0],:]

  stored_simulated_outputs_array = np.copy(stored_simulated_outputs_array_all)
  stored_simulated_inputs_con_sat_array = np.copy(stored_inputs_array_not_ohe)


  # In[]:


  # In[]: Get corner points
  corner_inputs, corner_outputs = getTopParetoFront(number_of_top_individuals=1)
  print('Corner Inputs are ', corner_inputs)
  print('Corner Outputs are ', corner_outputs)
  # best_input, best_output = getTopParetoFront(number_of_top_individuals=2)
  # In[]:
  # best_objective = max(stored_simulated_outputs_array[:,0])
  # best_objective_index = np.argmax(stored_simulated_outputs_array[:,0])
  # # best_input = stored_inputs_array[best_objective_index]
  # best_input_ohe = stored_simulated_inputs_con_sat_array[best_objective_index]
  # print('Best objective till now is ', best_objective)
  # print('Best input OHE till now is ', best_input_ohe)
  
  # sorted_indices =   np.argsort(-stored_simulated_outputs_array[:,0])
  # keep_data_len = 1000
  if (stored_simulated_outputs_array.shape[0]>keep_data_len):
      print('Clipping')
      top_1kInputs, top_1kOutputs = getTopParetoFront(number_of_top_individuals=keep_data_len)      
      # sorted_indices_subset = sorted_indices[:keep_data_len]  top_1kOutputs
      stored_simulated_outputs_array = np.array(top_1kOutputs)
      stored_simulated_inputs_con_sat_array_not_ohe = np.array(top_1kInputs)
      
      # material_choice = stored_simulated_inputs_con_sat_array_not_ohe[:,2].reshape(-1,1)
      # material_choice_ohe = enc.transform(material_choice)
      # stored_simulated_inputs_con_sat_array_not_ohe_copy = stored_simulated_inputs_con_sat_array_not_ohe[:,[0,1,3,4,5,6,7,8,9,10,11,12]]
      # stored_simulated_inputs_con_sat_array_not_ohe_colmuns_useful = np.copy(stored_simulated_inputs_con_sat_array_not_ohe_copy)
      # stored_simulated_inputs_con_sat_array = np.hstack((stored_simulated_inputs_con_sat_array_not_ohe_colmuns_useful, material_choice_ohe))       
      stored_simulated_inputs_con_sat_array = np.copy(stored_simulated_inputs_con_sat_array_not_ohe)
  print('Training data size is ', stored_simulated_outputs_array.shape)      
    
  # In[]:
  bo_inputs_raw_all = np.copy(stored_simulated_inputs_con_sat_array) 
  
  bo_outputs_raw_all = np.copy(stored_simulated_outputs_array)   
  # In[]:
  bo_outputs_raw_pos_indices = np.where(bo_outputs_raw_all[:,0]>0)  
  bo_inputs_raw = bo_inputs_raw_all[bo_outputs_raw_pos_indices[0]]
  bo_outputs_raw = bo_outputs_raw_all[bo_outputs_raw_pos_indices[0]]
  print('Positive reward size is ', bo_outputs_raw.shape[0])
  # In[]:
  bo_inputs_lower_bound_raw = lower_bound 
  bo_inputs_upper_bound_raw = upper_bound 
  print('Lower and upper bounds are ', lower_bound, upper_bound)
    
  scaler_input_bo = MinMaxScaler(feature_range=(0,1))
  # scaler_output_bo = StandardScaler()
  scaler_output_bo = MinMaxScaler(feature_range=(0,1))
  scaler_input_bo.fit(bo_inputs_raw)  
  scaler_output_bo.fit(bo_outputs_raw)   
  bo_inputs_raw_scaled = scaler_input_bo.transform(bo_inputs_raw)
  bo_outputs_raw_scaled = scaler_output_bo.transform(bo_outputs_raw) 
  
  bo_inputs_lower_bound_raw_scaled =  scaler_input_bo.transform(
                                          np.array(bo_inputs_lower_bound_raw).reshape(1,-1)).reshape(-1,).tolist()
  bo_inputs_upper_bound_raw_scaled =  scaler_input_bo.transform(
                                          np.array(bo_inputs_upper_bound_raw).reshape(1,-1)).reshape(-1,).tolist()  
    
  # In[OutputBounds]:
  bo_outputs_lower_bound_raw = [0]

  bo_outputs_upper_bound_raw = [1.1*np.max(stored_simulated_outputs_array_all[:,0]),
                                 ]                             
  # bo_outputs_lower_bound_raw  = [0,
  #                                0,
  #                                c1_limit, 
  #                                c2_limit,
  #                                c3_limit_lower, 
  #                                c4_limit_lower] 

  # c1_limit_upper = np.max(stored_simulated_outputs_array_all[:,1])
  # c2_limit_upper = np.max(stored_simulated_outputs_array_all[:,2])
  
  
  # bo_outputs_upper_bound_raw  = [best_output[0][0],
  #                                best_output[0][1],
  #                                c1_limit_upper, 
  #                                c2_limit_upper,
  #                                c3_limit_upper, 
  #                                c4_limit_upper] 
  # bo_outputs_upper_bound_raw  = [max(best_output[0][0],best_output[1][0]),
  #                                max(best_output[0][1], best_output[1][1]),
  #                                c1_limit_upper, 
  #                                c2_limit_upper,
  #                                c3_limit_upper, 
  #                                c4_limit_upper] 
  
  bo_outputs_lower_bound_raw_scaled =  scaler_output_bo.transform(
                                          np.array(bo_outputs_lower_bound_raw).reshape(1,-1)).reshape(-1,).tolist()
  bo_outputs_upper_bound_raw_scaled =  scaler_output_bo.transform(
                                          np.array(bo_outputs_upper_bound_raw).reshape(1,-1)).reshape(-1,).tolist()        
    
    
  # In[]: 
  best_mse, best_model, best_arch_till_now = getBestNNArchSklearn(bo_inputs_raw_scaled, 
                                                                  bo_outputs_raw_scaled,
                                                                  arch_to_try)      
  print('Architecture being used is ', best_arch_till_now)        
  predicted_values = best_model.predict(bo_inputs_raw_scaled)        
      
      
      
  # In[]:   
  if len(best_model.coefs_) == 4:
      model = Sequential()
      model.add(Dense(best_model.coefs_[0].shape[1], input_shape=(bo_inputs_raw_scaled.shape[1],)))
      model.add(Activation('relu'))
      model.add(Dense(best_model.coefs_[1].shape[1]))
      model.add(Activation('relu'))
      model.add(Dense(best_model.coefs_[2].shape[1]))
      model.add(Activation('relu'))
      model.add(Dense(best_model.coefs_[-1].shape[1]))
  else:
      model = Sequential()
      model.add(Dense(best_model.coefs_[0].shape[1], input_shape=(bo_inputs_raw_scaled.shape[1],)))
      model.add(Activation('relu'))
      model.add(Dense(best_model.coefs_[1].shape[1]))
  # Transfer the learnt model to Keras
  # Set the weights using what was learnt from sklearn
  iter1 = 0
  for layer in (model.layers):
      if layer.name[0:3]=='den':
#             print(layer.name)
          layer_default_wt = model.get_layer(layer.name).get_weights()
          wt = [best_model.coefs_[iter1].reshape(layer_default_wt[0].shape),
                best_model.intercepts_[iter1].reshape(layer_default_wt[1].shape)]
          model.get_layer(layer.name).set_weights(wt)
          iter1+=1
  predicted_keras_values = model.predict(bo_inputs_raw_scaled)
  improvement_required = np.random.uniform(low = 1e-5, high = 1e-4)*0
  des_out_raw = top_outputs[0]
  des_out_raw_increased = copy.copy(des_out_raw)
  des_out_raw_increased[0] = des_out_raw_increased[0]*(1+improvement_required)
  print('Single CNMA desired output is ', des_out_raw)
  print('Fraction improvement required is ', improvement_required)
  print('Single CNMA desired output increased is ', des_out_raw_increased)
  des_out = scaler_output_bo.transform(np.array(des_out_raw_increased).reshape(1,-1)).tolist()[0]
  cnma_solution = generateEachCNMACandidateSolution(model,
                                                                                       bo_inputs_lower_bound_raw_scaled,
                                                                                       bo_inputs_upper_bound_raw_scaled,
                                                                                       bo_outputs_lower_bound_raw_scaled,
                                                                                       bo_outputs_upper_bound_raw_scaled,
                                                                                       enc,
                                                                                       scaler_input_bo,
                                                                                       scaler_output_bo,
                                                                                       des_out)
  # In[]:
  print('CNMA candidate solution is ', cnma_solution)
  if cnma_solution == [False]:
    return False
  else:
    return cnma_solution
                                                                    


################################# Generate Marabou candidate solution #########3

# In[BOTORCH Candidates]: 
################################################################################################################

# In[]:

################################################################################################################
# In[]:
def genCandidateSolutions(top_10_individuals,top_10_outputs,number_of_req_per_ind_gmm, 
                                number_of_req_per_ind_parallel_cnma, 
                                number_of_individuals_required_standard_max_ent,
                          number_of_individuals_required_standard_qUCB, acq_type):
  
    global gmm_details, cnma_details, cnma_parallel_details, qucb_details, conEI_details, qMaxEnt_details
    if (acq_type == 'gmm_'):
        start_gmm_candidate_generation_time = time.time() 
        gmm_candidate_inputs, gmm_candidate_outputs =   generateGMMCandidateSolutions(top_10_individuals,top_10_outputs,
                                                                                      number_of_req_per_ind = 
                                                                                      number_of_req_per_ind_gmm)
        end_gmm_candidate_generation_time = time.time() 
        gmm_candidate_generation_time = end_gmm_candidate_generation_time - start_gmm_candidate_generation_time
        # gmm_details['can_sol_gen_time'] += gmm_candidate_generation_time
        # gmm_candidate_inputs_to_use = gmm_candidate_inputs[:number_of_individuals_required_standard_gmm]    
        return (acq_type, gmm_candidate_inputs, gmm_candidate_generation_time)
    if (acq_type == 'cnma_'):
        start_cnma_candidate_generation_time = time.time()
        cnma_simple_based_candidate = generateCNMACandidateSolutions() 
        end_cnma_candidate_generation_time = time.time()
        cnma_candidate_generation_time = end_cnma_candidate_generation_time - start_cnma_candidate_generation_time
        # cnma_details['can_sol_gen_time'] += cnma_candidate_generation_time
        return (acq_type, cnma_simple_based_candidate, cnma_candidate_generation_time)
    if (acq_type == 'cnmaParallel_'):
        # cnma_based_candidate = generateCNMACandidateSolutions() 
        start_cnma_parallel_candidate_generation_time = time.time()
        cnma_based_candidate = generateCNMACandidateSolutionsParallel(top_10_individuals ,
                                                                      top_10_outputs,
                                                                      number_of_req_per_ind = number_of_req_per_ind_parallel_cnma)
        end_cnma_parallel_candidate_generation_time = time.time()
        cnma_parallel_candidate_generation_time = end_cnma_parallel_candidate_generation_time - start_cnma_parallel_candidate_generation_time
        # cnma_parallel_details['can_sol_gen_time'] += cnma_parallel_candidate_generation_time
        return (acq_type, cnma_based_candidate, cnma_parallel_candidate_generation_time)    
      
    if (acq_type == 'marabou_'):
        # cnma_based_candidate = generateCNMACandidateSolutions() 
        start_cnma_parallel_candidate_generation_time = time.time()
        cnma_based_candidate = generateMarabouCandidateSolutionsParallel(top_10_individuals ,
                                                                      top_10_outputs,
                                                                      number_of_req_per_ind = number_of_req_per_ind_parallel_cnma)
        end_cnma_parallel_candidate_generation_time = time.time()
        cnma_parallel_candidate_generation_time = end_cnma_parallel_candidate_generation_time - start_cnma_parallel_candidate_generation_time
        # cnma_parallel_details['can_sol_gen_time'] += cnma_parallel_candidate_generation_time
        return (acq_type, cnma_based_candidate, cnma_parallel_candidate_generation_time) 
      
    if (acq_type == 'qMAxEntropy_'):
        start_qmaxEnt_candidate_generation_time = time.time()
        try:
         bo_torch_solution_MaxEnt = generateBOTorchCandidateSolutionMaxEntropy(number_candidates=number_of_individuals_required_standard_max_ent, 
                                                                               device_to_use = device ) 
        except:
         bo_torch_solution_MaxEnt = generateBOTorchCandidateSolutionMaxEntropy(number_candidates=number_of_individuals_required_standard_max_ent, 
                                                                              device_to_use = device_cpu )  
        end_qmaxEnt_candidate_generation_time = time.time() 
        qMaxEnt_candidate_generation_time = -start_qmaxEnt_candidate_generation_time + end_qmaxEnt_candidate_generation_time
        # qMaxEnt_details['can_sol_gen_time'] += qMaxEnt_candidate_generation_time
        return (acq_type, bo_torch_solution_MaxEnt, qMaxEnt_candidate_generation_time)

    if (acq_type == 'qUCB_'):
        start_qUCB_candidate_generation_time = time.time()
        try:
         bo_torch_solution_qUCB = generateBOTorchCandidateSolutionqUCB(number_candidates=number_of_individuals_required_standard_qUCB, 
                                                                               device_to_use = device ) 
        except:
         bo_torch_solution_qUCB = generateBOTorchCandidateSolutionqUCB(number_candidates=number_of_individuals_required_standard_qUCB, 
                                                                              device_to_use = device_cpu )  
        end_qUCB_candidate_generation_time = time.time()
        qUCB_candidate_generation_time = end_qUCB_candidate_generation_time - start_qUCB_candidate_generation_time
        # qucb_details['can_sol_gen_time'] += qUCB_candidate_generation_time
        return (acq_type, bo_torch_solution_qUCB, qUCB_candidate_generation_time)    
    if (acq_type == 'conEI_'):
        start_conEI_candidate_generation_time = time.time()
        bo_torch_solution_conEI = generateBayesianCandidateSolutionsBOTORCConEI(number_candidates=1, 
                                                                               device_to_use = device )  
        end_conEI_candidate_generation_time = time.time()
        conEI_candidate_generation_time = end_conEI_candidate_generation_time - start_conEI_candidate_generation_time
        # conEI_details['can_sol_gen_time'] += conEI_candidate_generation_time
        return (acq_type, bo_torch_solution_conEI,conEI_candidate_generation_time)  

# In[]:
def genCandidateSolutionsSerial(top_10_individuals,top_10_outputs,number_of_req_per_ind_gmm, 
                                number_of_req_per_ind_parallel_cnma, number_of_individuals_required_standard_max_ent,
                          number_of_individuals_required_standard_qUCB, acq_type):
    global gmm_details, cnma_details, cnma_parallel_details, qucb_details, conEI_details, qMaxEnt_details
    if (acq_type == 'gmm_'):
        start_gmm_candidate_generation_time = time.time() 
        gmm_candidate_inputs, gmm_candidate_outputs =   generateGMMCandidateSolutions(top_10_individuals,top_10_outputs,
                                                                                      number_of_req_per_ind = 
                                                                                      number_of_req_per_ind_gmm)
        end_gmm_candidate_generation_time = time.time() 
        gmm_candidate_generation_time = end_gmm_candidate_generation_time - start_gmm_candidate_generation_time
        # gmm_details['can_sol_gen_time'] += gmm_candidate_generation_time
        # gmm_candidate_inputs_to_use = gmm_candidate_inputs[:number_of_individuals_required_standard_gmm]    
        return (acq_type, gmm_candidate_inputs, gmm_candidate_generation_time)
    if (acq_type == 'cnma_'):
        start_cnma_candidate_generation_time = time.time()
        cnma_simple_based_candidate = generateCNMACandidateSolutions() 
        end_cnma_candidate_generation_time = time.time()
        cnma_candidate_generation_time = end_cnma_candidate_generation_time - start_cnma_candidate_generation_time
        # cnma_details['can_sol_gen_time'] += cnma_candidate_generation_time
        return (acq_type, cnma_simple_based_candidate, cnma_candidate_generation_time)
    if (acq_type == 'cnmaParallel_'):
        # cnma_based_candidate = generateCNMACandidateSolutions() 
        start_cnma_parallel_candidate_generation_time = time.time()
        cnma_based_candidate = generateCNMACandidateSolutionsParallel(top_10_individuals ,
                                                                      top_10_outputs,
                                                                      number_of_req_per_ind = number_of_req_per_ind_parallel_cnma)
        end_cnma_parallel_candidate_generation_time = time.time()
        cnma_parallel_candidate_generation_time = end_cnma_parallel_candidate_generation_time - start_cnma_parallel_candidate_generation_time
        # cnma_parallel_details['can_sol_gen_time'] += cnma_parallel_candidate_generation_time
        return (acq_type, cnma_based_candidate, cnma_parallel_candidate_generation_time)   
      
    if (acq_type == 'marabou_'):
        # cnma_based_candidate = generateCNMACandidateSolutions() 
        start_cnma_parallel_candidate_generation_time = time.time()
        cnma_based_candidate = generateMarabouCandidateSolutionsParallel(top_10_individuals ,
                                                                      top_10_outputs,
                                                                      number_of_req_per_ind = number_of_req_per_ind_parallel_cnma)
        end_cnma_parallel_candidate_generation_time = time.time()
        cnma_parallel_candidate_generation_time = end_cnma_parallel_candidate_generation_time - start_cnma_parallel_candidate_generation_time
        # cnma_parallel_details['can_sol_gen_time'] += cnma_parallel_candidate_generation_time
        return (acq_type, cnma_based_candidate, cnma_parallel_candidate_generation_time)  
      
    if (acq_type == 'qMAxEntropy_'):
        start_qmaxEnt_candidate_generation_time = time.time()
        try:
         bo_torch_solution_MaxEnt = generateBOTorchCandidateSolutionMaxEntropy(number_candidates=number_of_individuals_required_standard_max_ent, 
                                                                               device_to_use = device ) 
        except:
         bo_torch_solution_MaxEnt = generateBOTorchCandidateSolutionMaxEntropy(number_candidates=number_of_individuals_required_standard_max_ent, 
                                                                              device_to_use = device_cpu )  
        end_qmaxEnt_candidate_generation_time = time.time() 
        qMaxEnt_candidate_generation_time = -start_qmaxEnt_candidate_generation_time + end_qmaxEnt_candidate_generation_time
        # qMaxEnt_details['can_sol_gen_time'] += qMaxEnt_candidate_generation_time
        return (acq_type, bo_torch_solution_MaxEnt, qMaxEnt_candidate_generation_time)

    if (acq_type == 'qUCB_'):
        start_qUCB_candidate_generation_time = time.time()
        try:
         bo_torch_solution_qUCB = generateBOTorchCandidateSolutionqUCB(number_candidates=number_of_individuals_required_standard_qUCB, 
                                                                               device_to_use = device ) 
        except:
         bo_torch_solution_qUCB = generateBOTorchCandidateSolutionqUCB(number_candidates=number_of_individuals_required_standard_qUCB, 
                                                                              device_to_use = device_cpu )  
        end_qUCB_candidate_generation_time = time.time()
        qUCB_candidate_generation_time = end_qUCB_candidate_generation_time - start_qUCB_candidate_generation_time
        # qucb_details['can_sol_gen_time'] += qUCB_candidate_generation_time
        return (acq_type, bo_torch_solution_qUCB, qUCB_candidate_generation_time)    
    if (acq_type == 'conEI_'):
        start_conEI_candidate_generation_time = time.time()
        bo_torch_solution_conEI = generateBayesianCandidateSolutionsBOTORCConEI(number_candidates=1, 
                                                                               device_to_use = device )  
        end_conEI_candidate_generation_time = time.time()
        conEI_candidate_generation_time = end_conEI_candidate_generation_time - start_conEI_candidate_generation_time
        # conEI_details['can_sol_gen_time'] += conEI_candidate_generation_time
        return (acq_type, bo_torch_solution_conEI,conEI_candidate_generation_time)  

  # In[]:
  # concatenetae inputs and outputs

# In[New population]:

# In[]:
# In[New population]:

# In[Run GA!]:
# In[]:
# population size
pop_size = 100
# pop_size = 30
# Create 1st generation of individuals
population_all_uncorrected = create_population(individuals = pop_size)

population_all = population_all_uncorrected

# Maximum number of generations
number_generation = 400
# number of no_objectives
no_objectives = 1
# In[]:
# Keep track of the simulations which will be used later to train CNMA
number_of_components = len(lower_bound)
number_of_simulations = number_generation*pop_size
simulations_track_input = np.zeros(shape = (number_of_simulations,number_of_components ))
simulations_track_output = np.zeros(shape = (number_of_simulations,no_objectives ))

fitness_track_obj1 = np.zeros((number_generation,pop_size))
# fitness_track_obj2 = np.zeros((number_generation,pop_size))
# fitness_track_c1 = np.zeros((number_generation,pop_size))
# fitness_track_c2 = np.zeros((number_generation,pop_size))


# fitness_track_con= np.zeros((number_generation,pop_size))
# In[]:
best_obj_tracked = np.zeros((number_generation,))
all_inputs = []
all_outputs = []
ga_time_tracked = []
# In[]:
# fitness_track_fuel = np.zeros((number_generation,pop_size))
# fitness_track_time= np.zeros((number_generation,pop_size))
# fitness_track_reward= np.zeros((number_generation,pop_size))
# number_generation = 8
# In[]:
best_reward_tracked = np.zeros((number_generation,))
# In[]:
# In[]:
#TODO Parse power properly - not always at same location - use regular expression

no_improve_count = 0
improve_Count = 0
wrong_count = 0
this_simulation_number = 0
stopThreshold = 1000000 # threshold when to stop the simulation, if fitness falls below this, we can stop GA
stopsimulation = False
switched_generation_stamp = False
# switched_generation_number = 0
gen_since_last_improved = 0

# In[]:
number_generation=3
# In[]:
# time_out=100
for generation in tqdm(range(0,number_generation)):
    elapsed = time.time() - t
    ga_time_tracked.append(elapsed) 
    fitness_array = np.zeros((len(population_all), no_objectives))
    print('Unique simulations done till now is',  len(stored_inputs))
    print('--'*50)
    print('Unique simulations done till now is',  len(stored_simulated_outputs))
    if generation%10==0:
#      save every 10 generations
      filename = 'storeSim/rvdPolakVer1_' + str(datetime.now()) + str(generation) + '.pkl'
      # dill.dump_session(filename)
#    if generation%10==0:
##      save every 10 generations
#      filename = 'storeSim/rvdPolakVer1_' + str(datetime.now()) + str(generation) + '.pkl'
#      dill.dump_session(filename)
    if generation == 0:
      start_sim_time = time.time()
      for i,pop in tqdm(enumerate(population_all)):
        this_fitness = computeFitnesIndividual(pop)
        fitness_array[i,:] = np.array(this_fitness)
        this_simulation_number+=1
      total_simulation_time = time.time() - start_sim_time
      parallel_simulation_time_tracked.append(total_simulation_time)
      print('Total simulation time in parallel is', total_simulation_time)
    else:
        for i,pop in tqdm(enumerate(population_all)):
            fitness_array[i,:] = fitness_next_gen[i,:]
    all_inputs.append(population_all)
    all_outputs.append(fitness_array)

    fitness_track_obj1[generation,:] = fitness_array[:,0]
    # fitness_track_con[generation,:] = fitness_array[:,1]
    # fitness_track_reward[generation,:] = fitness_array[:,2]
#    fitness_track_components[generation,:] = fitness_array[:,2]
    print('generation')
    print(generation)
    if generation>1:
        sorted_fitnessCurrent = np.sort(fitness_track_obj1[generation,:])
        sorted_fitnessCurrent_index = np.argsort(fitness_track_obj1[generation,:])
        sorted_fitnessPrev = np.sort(fitness_track_obj1[generation-1,:])
        best_obj_tracked[generation-1] = sorted_fitnessCurrent[0]
        if (sorted_fitnessCurrent[0] < sorted_fitnessPrev[0]):
            print("say Improved!")
            improve_Count+=1
            print(improve_Count)
        if (sorted_fitnessCurrent[0] > sorted_fitnessPrev[0]):
            print("say Something is wrong!")
            wrong_count+=1
            print(improve_Count)
        if (sorted_fitnessCurrent[0] == sorted_fitnessPrev[0]):
#            os.system("say No Improvement!")
            no_improve_count+=1
            print(improve_Count)
        # if ((sorted_fitnessCurrent[0] < stopThreshold) and 
        #     (fitness_track_con[generation, sorted_fitnessCurrent_index[0]] < 10)):
        #   stopsimulation = True
    if generation>10:
      subarray_same_fitness = best_obj_tracked[generation-10:generation]
      print('subarray_same_fitness')
      print(subarray_same_fitness)
      difference_subarray_sum = np.sum(np.diff(subarray_same_fitness))
      print('difference_subarray_sum')
      print(difference_subarray_sum)
      print('Stopping due to consistent fitness')
      if difference_subarray_sum == 0:
        stopsimulation = True
      
           

    print(np.sort(fitness_array[:,0]))
    print(np.argsort(fitness_array[:,0]))
    if ((generation>1) and (stopsimulation)):
      break
    if generation<(number_generation-1):
        new_population, fitness_next_gen = generateNewPopulationNSGA(population_all, fitness_array ,number_elites = 5,
        tournament_size = 10, crossover_prob = 0.9, mutationn_prob = 0.1)
        population_all = new_population

    print('='*50)
# In[]:
step1_time_elapsed =copy.copy(elapsed)
step2_budget_left = complete_step1_step2_time_limit - step1_time_elapsed
# In[]:
print('=='*50)
print('Step 1 is done - going to Step 2')
print('=='*50)

# In[]:
def clipInputRangesWithinPercent(input_to_surround, clip_percent = 0.7):
  lower_bound_clip = []
  upper_bound_clip = []
  for bound_iter, inp in enumerate(input_to_surround):
    this_lb = inp*(1 - clip_percent)
    this_ub = inp*(1 + clip_percent)
      
    lower_bound_clip.append(this_lb)
    upper_bound_clip.append(this_ub)
  lower_bound_clip_l = clipInputsWithinRange(lower_bound_clip)
  upper_bound_clip_u = clipInputsWithinRange(upper_bound_clip)
  return lower_bound_clip_l, upper_bound_clip_u
# In[Step 2 here]:

# Generate samples around the best design
corner_inputs, corner_outputs = getTopParetoFront(number_of_top_individuals=1) 
lb_small, ub_small = clipInputRangesWithinPercent(corner_inputs[0], 0.01)
# lower_bound, upper_bound = clipInputRangesWithinPercent(corner_inputs[0], 0.7)
arr_list = []
for k in range(0,len(lower_bound)):
  lb = lb_small[k]
  ub = ub_small[k]
  this_ele = [lb, ub]
  arr_list.append(this_ele) 
# In[]:
max_samples = 100000
xlimits = np.array(arr_list)
sampling = LHS(xlimits = xlimits)
raw_samples = sampling(max_samples)

# In[]:
simulation_samples = raw_samples[:100].tolist()
for surr in simulation_samples:
  this_fitness = computeFitnesIndividual(surr)
# In[]:
global_sobol_counter = len(simulation_samples)
def create_individual_step2(current_try=0):
    global global_sobol_counter
#    available_components = list(parts.keys())
    print('Sampled Sobol individual is ')
    sobol_sample = raw_samples[global_sobol_counter,:]
    print(sobol_sample)
    global_sobol_counter+=1
    print('Total sobol samples are', global_sobol_counter)
    return sobol_sample.tolist()
# In[]:
  
       

step_2_calls = 0
budget_max_time_step2 = 3*3600
print('Time budget for Step 2 is ', budget_max_time_step2)
time_used_step_2 = 0




# In[]:
while (time_used_step_2<budget_max_time_step2):
  # choose the best acquisition function
  step_2_calls+=1
  if (step_2_calls%50==0):
    stored_inputs_array = np.concatenate(([stored_inputs]), axis = 0)
    stored_objectives_array = np.concatenate(([stored_objectives]), axis = 0)
    stored_simulated_outputs_array = np.concatenate(([stored_simulated_outputs]), axis = 0)
    filename = 'storeSim/LunarLandeRoboSysDes100Ind_' + str(datetime.now()) + '_' + str(step_2_calls) + '_.npz'
    
    np.savez(filename,
             stored_inputs_array = stored_inputs_array,
             stored_objectives_array = stored_objectives_array,
             stored_simulated_outputs_array = stored_simulated_outputs_array)
    
    print('Saved filename is ',filename)
    print('Total calls in Step 2 are ',step_2_calls)
    print('Total simulations done are ',stored_inputs_array.shape[0])  
    
  start_time_step2 = time.time()
  # best_acq_function = getBestAcquisitionFunction()
  # best_acq_function=0
  # start_can_sol_gen_time = time.time()
  # candidate_solutions_step2_are = generateCandidatesolution(best_acq_function)
  corner_inputs_old, corner_outputs_old = getTopParetoFront(number_of_top_individuals=1)   
  best_objective = corner_outputs_old[0][0]
  print('Best objective till now is ',best_objective)

  cnma_single_solution = generateSingleCNMACandidateSolution(corner_inputs_old, corner_outputs_old, 1)
  if cnma_single_solution == False:
    cnma_single_solution = create_individual_step2()
  else:
    cnma_single_solution = cnma_single_solution[0]
  this_fitness = computeFitnesIndividual(cnma_single_solution)

  corner_inputs, corner_outputs = getTopParetoFront(1)
  # bandwidth_constraint = corner_outputs[0][-1]
  end_time_step2 = time.time()
  time_used_in_this_part_step2 = end_time_step2 - start_time_step2
  time_used_step_2+=time_used_in_this_part_step2
  print('Time used in Step 2 is ', time_used_in_this_part_step2)










