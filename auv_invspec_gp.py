# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from auv.problem import AUVsim

# design optimization module
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)
from pymoo.core.duplicate import DefaultDuplicateElimination
from invspec.nsga_inv_spec import NSGAInvSpec

# human simulator module
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRankerWeights

# inverse specification module
from funct_approx.config import GPConfig
from invspec.inv_spec import InvSpec
from invspec.query_selector.random_selector import RandomQuerySelector
from invspec.query_selector.mutual_info_query_selector import (
    MutualInfoQuerySelector
)
from invspec.inference.reward_GP import RewardGP

# others
from utils import (
    set_seed, save_obj, get_infeasible_designs, get_inference_output,
    plot_result_3D, plot_result_pairwise, plot_output_2D, plot_output_3D,
    normalize
)
from config.config import load_config
from shutil import copyfile


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_ga = config_dict['GA']
  config_inv_spec = config_dict['INV_SPEC']
  config_gp = config_dict['GP']
  config_human = config_dict['HUMAN']

  out_folder = os.path.join(
      'scratch', 'auv_' + config_general.PROBLEM_TYPE, 'invspec_gp',
      config_human.TYPE + '_' + config_inv_spec.QUERY_SELECTOR_TYPE,
      config_general.NAME
  )
  print(out_folder)
  fig_folder = os.path.join(out_folder, 'figure')
  os.makedirs(fig_folder, exist_ok=True)
  agent_folder = os.path.join(out_folder, 'agent')
  os.makedirs(agent_folder, exist_ok=True)
  obj_eval_folder = os.path.join(out_folder, 'obj_eval')
  os.makedirs(obj_eval_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))

  # endregion

  # region: == local functions ==
  def getHumanScores(design_feature, w_opt, active_constraint_set=None):
    n_designs = design_feature.shape[0]
    indicator = get_infeasible_designs(design_feature, active_constraint_set)
    feasible_index = np.arange(n_designs)[np.logical_not(indicator)]
    feasible_designs = design_feature[feasible_index]
    scores = feasible_designs @ w_opt
    return feasible_index, scores

  # endregion

  # region: == Define Problem ==
  set_seed(seed_val=config_general.SEED, use_torch=True)
  problem = AUVsim(problem_type=config_general.PROBLEM_TYPE)
  n_obj = problem.n_obj
  objective_names = problem.fparams.func.objective_names
  print(problem)
  print('inputs:', problem.fparams.xinputs)

  objectives_bound = np.empty(shape=(3, 2))
  objectives_bound[:, 0] = np.array([0., 0.4, -350.])
  objectives_bound[:, 1] = np.array([12000., 1.6, 0.])
  # endregion

  # region: == Define Algorithm ==
  sampling = MixedVariableSampling(
      problem.fparams.mask, {
          "real": get_sampling("real_random"),
          "int": get_sampling("int_random")
      }
  )

  crossover = MixedVariableCrossover(
      problem.fparams.mask, {
          "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
          "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
      }
  )

  mutation = MixedVariableMutation(
      problem.fparams.mask, {
          "real": get_mutation("real_pm", eta=3.0),
          "int": get_mutation("int_pm", eta=3.0)
      }
  )

  algorithm = NSGAInvSpec(
      pop_size=config_ga.POP_SIZE, n_offsprings=config_ga.POP_SIZE,
      sampling=sampling, crossover=crossover, mutation=mutation,
      eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-3),
      warmup=config_inv_spec.NUM_WARMUP, beta=config_inv_spec.BETA
  )

  # termination criterion
  numGenTotal = config_ga.NUM_GEN
  termination = get_termination("n_gen", numGenTotal)
  # endregion

  # region: == Define Human Simulator ==
  print("\n== Human Simulator ==")
  active_constraint_set = None
  if config_human.TYPE == 'speed':
    w_opt = np.array([0.1, 0.7, 0.2])
  else:
    w_opt = np.array([0.5, 0.1, 0.4])
    if config_human.TYPE == 'range_hard':
      active_constraint_set = [['0', 0.2], ['1', 0.2]]
  print("Human simulator has weights below")
  print(w_opt)
  if active_constraint_set is not None:
    print("Human simulator has active constraints:")
    print(active_constraint_set)

  human = HumanSimulator(
      ranker=PairRankerWeights(
          w_opt, beta=config_human.BETA,
          active_constraint_set=active_constraint_set,
          perfect_rank=config_human.PERFECT_RANK
      )
  )
  # endregion

  # region: == Define invSpec ==
  print("\n== InvSpec Construction ==")
  CONFIG = GPConfig(
      SEED=config_general.SEED,
      HORIZONTAL_LENGTH=config_gp.HORIZONTAL_LENGTH,
      VERTICAL_VARIATION=config_gp.VERTICAL_VARIATION,
      NOISE_LEVEL=config_gp.NOISE_LEVEL,
      BETA=config_inv_spec.BETA,
  )
  print(vars(CONFIG), '\n')

  if config_inv_spec.POP_EXTRACT_TYPE == 'F':
    dimension = problem.n_obj
    input_normalize = True
    input_min = objectives_bound[:, 0]
    input_max = objectives_bound[:, 1]
  elif config_inv_spec.POP_EXTRACT_TYPE == 'X':
    dimension = problem.n_var
    input_normalize = False
    input_min = None
    input_max = None
  else:
    raise ValueError(
        "The pop_extract_type ({}) is not supported".format(
            config_inv_spec.POP_EXTRACT_TYPE
        )
    )
  initial_point = np.zeros(dimension)

  if config_inv_spec.QUERY_SELECTOR_TYPE == 'rand':
    agent = InvSpec(
        inference=RewardGP(
            dimension, 0, CONFIG, initial_point, input_min=input_min,
            input_max=input_max, input_normalize=input_normalize,
            pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
        ), query_selector=RandomQuerySelector()
    )
  else:
    agent = InvSpec(
        inference=RewardGP(
            dimension, 0, CONFIG, initial_point, input_min=input_min,
            input_max=input_max, input_normalize=input_normalize,
            pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
        ), query_selector=MutualInfoQuerySelector()
    )
  # endregion

  # region: == Optimization ==
  print("\n== Optimization starts ...")
  # perform a copy of the algorithm to ensure reproducibility
  obj = copy.deepcopy(algorithm)

  # let the algorithm know what problem we are intending to solve and provide
  # other attributes
  obj.setup(
      problem, termination=termination, seed=config_general.SEED,
      save_history=True
  )

  # until the termination criterion has not been met
  agent.clear_feedback()
  updateTimes = 0
  start_time = time.time()
  while obj.has_next():
    #= perform an iteration of the algorithm
    obj.next()
    n_acc_fb = agent.get_number_feedback()

    #= check performance
    if obj.n_gen % config_general.CHECK_GEN == 0:
      n_gen = obj.n_gen
      n_nds = len(obj.opt)
      CV = obj.opt.get('CV').min()
      print(f"gen[{n_gen}]: n_nds: {n_nds} CV: {CV}")
      features = -obj.opt.get('F')
      if n_obj == 3:
        fig = plot_result_3D(features, objective_names, objectives_bound)
      else:
        fig = plot_result_pairwise(
            n_obj, features, objective_names, objectives_bound
        )
      fig.supxlabel(
          'G{}: {} cumulative queries'.format(n_gen, n_acc_fb), fontsize=20
      )
      fig.tight_layout()
      fig_progress_folder = os.path.join(fig_folder, 'progress')
      os.makedirs(fig_progress_folder, exist_ok=True)
      fig.savefig(os.path.join(fig_progress_folder, str(n_gen) + '.png'))
      plt.close()

      features_normalized = normalize(
          -obj.pop.get('F'), input_min=objectives_bound[:, 0],
          input_max=objectives_bound[:, 1]
      )
      feasible_index, scores = getHumanScores(
          features_normalized, w_opt, active_constraint_set
      )

      res_dict = dict(
          features=features_normalized, component_values=obj.pop.get('X'),
          scores=scores, feasible_index=feasible_index
      )
      obj_eval_path = os.path.join(obj_eval_folder, 'obj' + str(obj.n_gen))
      save_obj(res_dict, obj_eval_path)

    #= interact with human
    if config_inv_spec.NUM_WARMUP == 0:
      time2update = ((obj.n_gen == 1)
                     or (obj.n_gen % config_inv_spec.INTERACT_PERIOD == 0))
    else:
      time2update = (
          obj.n_gen - config_inv_spec.NUM_WARMUP
      ) % config_inv_spec.INTERACT_PERIOD == 0
    if time2update and (obj.n_gen < numGenTotal):
      print("\nAt generation {}".format(obj.n_gen))
      features = normalize(
          -obj.pop.get('F'), input_min=objectives_bound[:, 0],
          input_max=objectives_bound[:, 1]
      )  # we want to maximize
      components = obj.pop.get('X')

      n_want = config_inv_spec.MAX_QUERIES_PER
      if n_acc_fb + n_want > config_inv_spec.MAX_QUERIES:
        n_ask = n_want - n_acc_fb
      else:
        n_ask = n_want

      if n_ask > 0:
        # 1. pick and send queries to humans
        indices = agent.get_query(obj.opt, n_ask)

        # 2. get feedback from humans
        for idx in indices:
          query_features = features[idx, :]
          query_components = components[idx, :]
          query = dict(F=query_features, X=query_components)
          fb_raw = human.get_ranking(query)

          if config_inv_spec.POP_EXTRACT_TYPE == 'F':
            q_1 = (query_features[0:1, :], np.array([]).reshape(1, 0))
            q_2 = (query_features[1:2, :], np.array([]).reshape(1, 0))
          else:
            q_1 = (query_components[0:1, :], np.array([]).reshape(1, 0))
            q_2 = (query_components[1:2, :], np.array([]).reshape(1, 0))

          if fb_raw == 0:
            fb_invspec = 1
          elif fb_raw == 1:
            fb_invspec = -1
          elif fb_raw == 2:
            eps = np.random.uniform()
            fb_invspec = 1 if eps > 0.5 else -1
          agent.store_feedback(q_1, q_2, fb_invspec)

        n_fb = len(indices)
        n_acc_fb = agent.get_number_feedback()
        print(
            "Collect {:d} feedback, Accumulated {:d} feedback".format(
                n_fb, n_acc_fb
            )
        )

        # 3. update fitness function
        _ = agent.learn()
        obj.fitness_func = agent.inference
        updateTimes += 1

        # 4. store and report
        indices = np.argsort(features[:, 0])
        features = features[indices]
        save_obj(agent, os.path.join(agent_folder, 'agent' + str(updateTimes)))

        feasible_index, scores = getHumanScores(
            features, w_opt, active_constraint_set
        )
        acc = len(feasible_index) / len(features)
        print('Feasible ratio: {:.3f}'.format(acc))
        with np.printoptions(formatter={'float': '{: .3f}'.format}):
          print(scores)
        print()
      else:
        print("Exceed maximal number of queries!", end=' ')
        print("Accumulated {:d} feedback".format(n_acc_fb))

  end_time = time.time()
  print("It took {:.1f} seconds".format(end_time - start_time))
  # endregion

  # region: finally obtain the result object
  save_obj(agent, os.path.join(out_folder, 'agent'))

  res = obj.result()
  if config_general.USE_TIMESTR:
    timestr = time.strftime("%m-%d-%H_%M")
  else:
    timestr = ''
  pickle_path = os.path.join(out_folder, timestr + 'res.pkl')
  with open(pickle_path, 'wb') as output:
    pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

  features = -res.F
  fig = plot_result_pairwise(
      n_obj, features, objective_names, objectives_bound,
      active_constraint_set=active_constraint_set
  )
  fig.tight_layout()
  fig.savefig(os.path.join(fig_folder, 'obj_pairwise.png'))

  indices = np.argsort(features[:, 0])
  features = features[indices]
  _features = normalize(
      features,
      input_min=objectives_bound[:, 0],
      input_max=objectives_bound[:, 1],
  )
  obj_eval_path = os.path.join(
      obj_eval_folder, 'obj' + str(updateTimes) + '.npy'
  )
  np.save(obj_eval_path, _features)
  feasible_index, scores = getHumanScores(
      _features, w_opt, active_constraint_set
  )
  acc = len(feasible_index) / len(_features)
  print()
  print('Feasible ratio: {:.3f}'.format(acc))
  with np.printoptions(formatter={'float': '{: .3f}'.format}):
    print(_features)
    print(scores)

  #== plot output ==
  if config_inv_spec.POP_EXTRACT_TYPE == 'F':
    obj_list_un = np.array([-150, -1])
    obj_list = ((obj_list_un - objectives_bound[2, 0]) /
                (objectives_bound[2, 1] - objectives_bound[2, 0]))
    level_ratios = np.array([0.25, 0.75])

    # plot hyper-parameters
    subfigsz = 4
    cm = 'coolwarm'
    lw = 2.5

    # get output
    nx = 101
    ny = 101
    X, Y, Z = get_inference_output(agent, nx, ny, obj_list)

    # 3D-plot
    fig = plot_output_3D(
        X, Y, Z, obj_list_un, objectives_bound, fsz=16, subfigsz=subfigsz,
        cm=cm
    )
    fig.tight_layout()
    fig_path = os.path.join(
        fig_folder, 'GP_op_' + config_inv_spec.QUERY_SELECTOR_TYPE + '.png'
    )
    fig.savefig(fig_path)

    # 2D-plot
    fig2, axes = plot_output_2D(
        X, Y, Z, obj_list_un, objectives_bound, level_ratios, fsz=18,
        subfigsz=subfigsz, cm=cm, lw=lw
    )
    for ax in axes:
      # plot constraint threshold
      ax.plot([0.2, 0.2], [0, 1], 'r:', lw=lw - 0.5)
      ax.plot([0, 1], [0.2, 0.2], 'r:', lw=lw - 0.5)
    fig2.tight_layout()
    fig_path = os.path.join(
        fig_folder, 'GP_op_2D_' + config_inv_spec.QUERY_SELECTOR_TYPE + '.png'
    )
    fig2.savefig(fig_path)
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "auv_invspec_gp.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
