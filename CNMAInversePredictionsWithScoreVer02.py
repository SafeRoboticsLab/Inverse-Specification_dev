#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 00:41:33 2021

@author: pterway
"""
# In[]
import pickle
input_file = 'kaichiehDataset/design_collections.pkl'
with open(input_file, 'rb') as input:
  designs_collection = pickle.load(input)
# designs_collection = pickle.load(input_file)
# In[]
component_values = designs_collection['component_values']
features = designs_collection['features']
scores = designs_collection['scores']
predicted_scores = designs_collection['predicted_scores']
dist2cluster_centers = designs_collection["dist2cluster_centers"]
# In[]

import sys
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import copy
# import pygmo as pg
from tqdm import tqdm
# import runcommand
# import pandas as pd
import os
import sobol_seq
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
# In[]
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/nsverify")
sys.path.append("/della/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits")
sys.path.append("/scratch/gpfs/pterway/HSPICE/HSPICE/MitCircuits/")
# In[]
from gurobipy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dense
from sklearn.neural_network import MLPRegressor
from nsverify import NetworkModel
from sklearn.preprocessing import MinMaxScaler
# In[]
num_cores = multiprocessing.cpu_count()
lower_bound = [0, 0, 0, 0, 0, 0]
upper_bound = [5, 5, 5, 5, 5, 40]
# In[Train keras model]:
def trainSklearnmodel(normalized_inputs, normalized_outputs, arch_to_use):
  X_train, X_test = normalized_inputs, normalized_outputs

  clf = MLPRegressor(hidden_layer_sizes=arch_to_use,
                      solver='adam', verbose=0, tol=1e-20,
                      learning_rate_init=.0001,  learning_rate='adaptive', max_iter=100000)       
  clf.fit(X_train, X_test)  
  loss = clf.loss_
  return loss, clf, arch_to_use

# In[dic from ind]:
best_mse, best_model, best_arch = None, None, None
nn_update_frequency = 50
arch_to_try = [(100), (15), (40,20,8), (20,20,8)] 
step_2_calls = 0

# In[]
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
# In[]
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
# In[]
def generateEachCNMACandidateSolution(model,
                                           bo_inputs_lower_bound_raw_scaled,
                                           bo_inputs_upper_bound_raw_scaled,
                                           bo_outputs_lower_bound_raw_scaled,
                                           bo_outputs_upper_bound_raw_scaled, 
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
    var_number +=1
 
  gmodel.update()
  inputs_model = input_gurobi_variables_g   
    
  # In[]:    
  constraints_NN_output = wrapper_gmodel.add_constraints(model.layers,
                                                  inputs_model,
                                                  dense_mod, relu_mod)    
    
    
  # In[]:
  cnma_obj1 = constraints_NN_output[0]   
  cnma_obj2 = constraints_NN_output[1]  
  cnma_obj3 = constraints_NN_output[2]  
  cnma_obj4 = constraints_NN_output[3]  

  
  # In[]:
  lbounds = des_out[0]
  ubounds = des_out[1]
  # In[]:
  cnma_obj1.lb = lbounds[0]
  cnma_obj2.lb = lbounds[1]
  cnma_obj3.lb = lbounds[2]
  cnma_obj4.lb = lbounds[3]

  cnma_obj1.ub = ubounds[0]
  cnma_obj2.ub = ubounds[1]
  cnma_obj3.ub = ubounds[2]   
  cnma_obj4.ub = ubounds[3]   
      
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
    # elapsed = time.time() - t 
    print('Feasible solution!!')
    
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
  NN_inputs = np.array(suggested_inputs_gb)
  raw_inputs = scaler_input_bo.inverse_transform(NN_inputs.reshape(1,-1)).reshape(-1,)
  raw_inputs_list = [raw_inputs.tolist()]      
  # In[]:
  estimated_inputs_array_denormalized_list_clipped = []
  for est_in_gmm in raw_inputs_list:
      clipped_input_to_range = clipInputsWithinRange(est_in_gmm)
      
      estimated_inputs_array_denormalized_list_clipped.append(clipped_input_to_range)
  # In[]:
  return estimated_inputs_array_denormalized_list_clipped
   

# In[]
# In[]:
def generateSingleCNMACandidateSolution(all_raw_inputs, all_raw_outputs, predicted_scores_till_now, desired_output_list):
  # In[]:
    
  print('Generating CNMA candidate solutions')
  stored_inputs_array_not_ohe = np.array(all_raw_inputs)
  
  stored_simulated_outputs_cluster_array = np.array(all_raw_outputs)
  predicted_scores_till_array = np.copy(predicted_scores_till_now)
  stored_simulated_outputs_array_all = np.concatenate((stored_simulated_outputs_cluster_array, predicted_scores_till_array), axis = 1)
  
  
  unique_gmm_inputs, ubique_gmm_inputs_indices = np.unique(
                        np.array(stored_inputs_array_not_ohe.tolist()), return_index=True, return_inverse=False, return_counts=False, axis=0)
  unique_gmm_outputs = stored_simulated_outputs_array_all[ubique_gmm_inputs_indices,:]  
  
  stored_simulated_outputs_array_all = np.copy(unique_gmm_outputs)
  stored_inputs_array_not_ohe = np.copy(unique_gmm_inputs)
  
  stored_simulated_outputs_array = np.copy(stored_simulated_outputs_array_all)
  stored_simulated_inputs_con_sat_array = np.copy(stored_inputs_array_not_ohe)


  # In[]:
    
  # In[]:
  bo_inputs_raw = np.copy(stored_simulated_inputs_con_sat_array) 
  
  bo_outputs_raw = np.copy(stored_simulated_outputs_array)   
  # In[]:
  bo_inputs_lower_bound_raw = copy.copy(lower_bound) 
  bo_inputs_upper_bound_raw = copy.copy(upper_bound) 
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
  bo_outputs_lower_bound_raw = [0, 0, 0, 0]

  bo_outputs_upper_bound_raw = [np.max(stored_simulated_outputs_array_all[:,0]),
                                 np.max(stored_simulated_outputs_array_all[:,1]),
                                 np.max(stored_simulated_outputs_array_all[:,2]),
                                 np.max(stored_simulated_outputs_array_all[:,3]),]                             

  
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
          
  # In[]
  des_out_raw_lb = desired_output_list[0]
  des_out_raw_ub = desired_output_list[1]
  
  # In[preprocess lower bounds]
  des_out_raw_lb_processed = []
  for kk, dol in enumerate(des_out_raw_lb):
    proc_out = copy.copy(dol)
    if dol==None:
      proc_out = bo_outputs_lower_bound_raw[kk]
    des_out_raw_lb_processed.append(proc_out)
  # In[preprocess upper bounds]
  des_out_raw_ub_processed = []
  for kk, dou in enumerate(des_out_raw_ub):
    proc_out = copy.copy(dou)
    if dou==None:
      proc_out = bo_outputs_upper_bound_raw[kk]
    des_out_raw_ub_processed.append(proc_out)   
  # In[]
  print('Desired outputs range are', [des_out_raw_lb_processed,des_out_raw_ub_processed] )
  des_out_raw_lb_processed_scaled = scaler_output_bo.transform(np.array(des_out_raw_lb_processed).reshape(1,-1)).tolist()[0]
  des_out_raw_ub_processed_scaled = scaler_output_bo.transform(np.array(des_out_raw_ub_processed).reshape(1,-1)).tolist()[0]
  des_out = [des_out_raw_lb_processed_scaled,
             des_out_raw_ub_processed_scaled]
  # In[]
  

  cnma_solution = generateEachCNMACandidateSolution(model,
                                                                                       bo_inputs_lower_bound_raw_scaled,
                                                                                       bo_inputs_upper_bound_raw_scaled,
                                                                                       bo_outputs_lower_bound_raw_scaled,
                                                                                       bo_outputs_upper_bound_raw_scaled,
                                                                                       scaler_input_bo,
                                                                                       scaler_output_bo,
                                                                                       des_out)
  # In[]:
  print('CNMA candidate solution is ', cnma_solution)
  if cnma_solution == [False]:
    return False
  else:
    return cnma_solution
# In[]
# Test Examplecnma_prediction
all_raw_inputs = np.copy(component_values)

all_raw_outputs = np.copy(dist2cluster_centers)
predicted_scores_till_now = np.copy(predicted_scores)


desired_output_list = [[20,1960,3280, .9],[26,1966,3288, 1.1]]
# desired_output_list = [[20,1960,None, .9],[26,1966,3288, 1.1]]

cnma_prediction = generateSingleCNMACandidateSolution(all_raw_inputs, all_raw_outputs,predicted_scores_till_now, desired_output_list)
print('Predicted input by cnma is ', cnma_prediction)