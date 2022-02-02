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

from swri.problem import SWRIProblem, SWRISimParallel
from swri.utils import report_pop_swri

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
from humansim.ranker.pair_ranker import PairRankerSimulator

# inverse specification module
from funct_approx.config import GPConfig
from invspec.inv_spec import InvSpec
from invspec.querySelector.random_selector import RandomQuerySelector
from invspec.querySelector.mutual_info_query_selector import (
    MutualInfoQuerySelector
)
from invspec.inference.reward_GP import RewardGP

# others
from utils import set_seed, save_obj, plot_result_pairwise, normalize
from config.config import load_config
from shutil import copyfile


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_ga = config_dict['GA']
  config_inv_spec = config_dict['INV_SPEC']
  config_gp = config_dict['GP']
  config_human = config_dict['HUMAN']

  out_folder = os.path.join('scratch', 'swri', 'invspec_gp')
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  fig_folder = os.path.join(out_folder, 'figure')
  os.makedirs(fig_folder, exist_ok=True)
  agent_folder = os.path.join(out_folder, 'agent')
  os.makedirs(agent_folder, exist_ok=True)
  obj_eval_folder = os.path.join(out_folder, 'obj_eval')
  os.makedirs(obj_eval_folder, exist_ok=True)
  fig_progress_folder = os.path.join(fig_folder, 'progress')
  os.makedirs(fig_progress_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))

  timestr = time.strftime("%m-%d-%H_%M")

  # endregion

  # region: == Define Problem ==
  print("\n== Problem ==")
  set_seed(seed_val=config_general.SEED, use_torch=True)
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblem(
      TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS,
      prefix="eval_" + time.strftime("%m-%d-%H_%M") + "_"
  )

  n_obj = problem.n_obj
  objective_names = problem.objective_names
  print('objectives', objective_names)
  print('inputs:', problem.input_names)

  x = np.array([[
      3.9971661079507594, 3.6711272495701843, 3.3501992857774856,
      3.0389318577493087, 4.422413267471787, 17.
  ]])
  y = {}
  problem._evaluate(x, y)
  print("\nGet the output from the problem:")
  print(y['F'])
  y = {}
  problem._evaluate(x, y)
  print(y['scores'])

  objectives_bound = np.array([
      [0, 4000],
      [-400, 0],
      [0, 30],
      [-50, 0.],
      [-12, 0.],
  ])
  scores_bound = np.array([-1e-8, 430])
  input_names_dict = {}
  for i in range(len(problem.input_names)):
    input_names_dict['o' + str(i + 1)] = problem.input_names[i][8:]
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  # endregion

  # region: == Define GA ==
  sampling = MixedVariableSampling(
      problem.input_mask, {
          "real": get_sampling("real_random"),
          "int": get_sampling("int_random")
      }
  )

  crossover = MixedVariableCrossover(
      problem.input_mask, {
          "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
          "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
      }
  )

  mutation = MixedVariableMutation(
      problem.input_mask, {
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
  if config_human.TYPE == 'has_const':
    active_constraint_set = [['0', 0.2], ['1', 0.2]]
  if active_constraint_set is not None:
    print("Human simulator has active constraints:")
    print(active_constraint_set)

  human = HumanSimulator(
      ranker=PairRankerSimulator(
          simulator=SWRISimParallel(
              TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS,
              prefix='human_'
          ), beta=config_human.BETA,
          active_constraint_set=active_constraint_set,
          perfect_rank=config_human.PERFECT_RANK,
          indifference=config_human.INDIFFERENCE
      )
  )
  print(human.ranker._get_scores(query=dict(X=x), feasible_index=[0]))
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
  elif config_inv_spec.POP_EXTRACT_TYPE == 'X':
    dimension = problem.n_var
  else:
    raise ValueError(
        "The input type ({}) of inv_spec is not supported".format(
            config_inv_spec.POP_EXTRACT_TYPE
        )
    )
  initial_point = np.zeros(dimension)
  if config_inv_spec.INPUT_NORMALIZE:
    input_normalize = True
    if config_inv_spec.POP_EXTRACT_TYPE == 'F':
      input_bound = objectives_bound
    elif config_inv_spec.POP_EXTRACT_TYPE == 'X':
      input_bound = component_values_bound
    input_min = input_bound[:, 0]
    input_max = input_bound[:, 1]
  else:
    input_normalize = False
    input_bound = None
    input_min = None
    input_max = None

  if config_inv_spec.QUERY_SELECTOR_TYPE == 'rand':
    agent = InvSpec(
        inference=RewardGP(
            dimension, 0, CONFIG, initial_point, input_min=input_min,
            input_max=input_max, input_normalize=input_normalize,
            pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
        ), querySelector=RandomQuerySelector()
    )
  else:
    agent = InvSpec(
        inference=RewardGP(
            dimension, 0, CONFIG, initial_point, input_min=input_min,
            input_max=input_max, input_normalize=input_normalize,
            pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
        ), querySelector=MutualInfoQuerySelector()
    )
  # endregion

  # region: == Optimization ==
  print("\n== Optimization starts ...")
  # perform a copy of the algorithm to ensure reproducibility
  obj = copy.deepcopy(algorithm)
  # obj.fitness_func = lambda x: 1.

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
  designs_collection = {}
  collect_undistinguished = False
  while obj.has_next():
    #= perform an iteration of the algorithm
    obj.next()
    print(obj.n_gen, end='\r')
    if obj.n_gen == config_inv_spec.NUM_WARMUP:
      designs_collection['component_values'] = obj.pop.get('X')
      designs_collection['features'] = -obj.pop.get('F')
      designs_collection['scores'] = obj.pop.get('scores')
    n_acc_fb = agent.get_number_feedback()

    #= check performance
    if obj.n_gen % config_general.CHECK_GEN == 0:
      report_pop_swri(
          obj, fig_progress_folder, n_acc_fb, objective_names,
          input_names_dict, objectives_bound, scores_bound,
          component_values_bound
      )

    #= interact with human
    if config_inv_spec.NUM_WARMUP == 0:
      time2update = ((obj.n_gen == 1)
                     or (obj.n_gen % config_inv_spec.INTERACT_PERIOD == 0))
    else:
      num_after_warmup = obj.n_gen - config_inv_spec.NUM_WARMUP
      time2update = (
          num_after_warmup % config_inv_spec.INTERACT_PERIOD == 0
          and num_after_warmup >= 0
      )

    if time2update and (obj.n_gen < numGenTotal):
      print("\nAt generation {}".format(obj.n_gen))
      features_unnormalized = -obj.pop.get('F')
      if config_inv_spec.INPUT_NORMALIZE:
        features = normalize(
            features_unnormalized, input_min=input_bound[:, 0],
            input_max=input_bound[:, 1]
        )  # we want to maximize
      else:
        features = features_unnormalized
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
        n_fb = 0
        for idx in indices:
          query_features = features[idx, :]
          query_components = components[idx, :]
          query = dict(F=query_features, X=query_components)
          fb_raw = human.get_ranking(query)
          print(fb_raw)

          if config_inv_spec.POP_EXTRACT_TYPE == 'F':
            q_1 = (query_features[0:1, :], np.array([]).reshape(1, 0))
            q_2 = (query_features[1:2, :], np.array([]).reshape(1, 0))
          else:
            q_1 = (query_components[0:1, :], np.array([]).reshape(1, 0))
            q_2 = (query_components[1:2, :], np.array([]).reshape(1, 0))

          if fb_raw != 2:
            n_fb += 1
            if fb_raw == 0:
              fb_invspec = 1
            elif fb_raw == 1:
              fb_invspec = -1
            agent.store_feedback(q_1, q_2, fb_invspec)
          elif collect_undistinguished:
            n_fb += 1
            eps = np.random.uniform()
            fb_invspec = 1 if eps > 0.5 else -1
            agent.store_feedback(q_1, q_2, fb_invspec)

        n_acc_fb = agent.get_number_feedback()
        print(
            "Collect {:d} feedback out of {:d} queries".format(n_fb, n_ask),
            "Accumulated {:d} feedback".format(n_acc_fb)
        )

        # 3. update fitness function
        _ = agent.learn()
        obj.fitness_func = agent.inference
        updateTimes += 1

        # 4. store and report
        indices = np.argsort(features[:, 0])
        features = features[indices]
        save_obj(agent, os.path.join(agent_folder, 'agent' + str(updateTimes)))
        obj_eval_path = os.path.join(
            obj_eval_folder, 'obj' + str(updateTimes - 1) + '.npy'
        )
        np.save(obj_eval_path, features)
        print()
      else:
        print("Exceed maximal number of queries!", end=' ')
        print("Accumulated {:d} feedback".format(n_acc_fb))

  end_time = time.time()
  print("It took {:.1f} seconds".format(end_time - start_time))
  # endregion

  # region: finally obtain the result object
  if "features" in designs_collection:  # have designs after GA
    designs_collection['component_values'] = np.concatenate(
        (obj.pop.get('X'), designs_collection['component_values'])
    )
    designs_collection['features'] = np.concatenate(
        (-obj.pop.get('F'), designs_collection['features'])
    )
    designs_collection['scores'] = np.concatenate(
        (obj.pop.get('scores'), designs_collection['scores'])
    )
  if agent.inference.input_normalize:
    features_final = agent.inference.normalize(designs_collection['features'])
  else:
    features_final = designs_collection['features']

  designs_collection['predicted_scores'] = np.empty(
      shape=designs_collection['scores'].shape
  )
  for i, feature in enumerate(features_final):
    designs_collection['predicted_scores'][i] = agent.inference.eval(feature)

  pickle_path = os.path.join(out_folder, 'design_collections.pkl')
  with open(pickle_path, 'wb') as output:
    pickle.dump(designs_collection, output, pickle.HIGHEST_PROTOCOL)
  print(designs_collection['scores'])
  print(designs_collection['predicted_scores'])

  res = obj.result()
  res_to_save = dict(X=res.X, F=res.F, pop=res.pop, opt=res.opt)
  pickle_path = os.path.join(out_folder, timestr + '.pkl')
  with open(pickle_path, 'wb') as output:
    pickle.dump(res_to_save, output, pickle.HIGHEST_PROTOCOL)
  print(pickle_path)

  features = -res.F
  fig = plot_result_pairwise(
      n_obj, features, objective_names, axis_bound=objectives_bound,
      n_col_default=5, subfigsz=4, fsz=16, sz=20
  )
  fig.tight_layout()
  fig.savefig(os.path.join(fig_folder, 'obj_pairwise.png'))

  print("\npick the design in the optimal front that has maximal objective 1.")
  indices = np.argsort(features[:, 0])
  features = features[indices]
  with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
    print(features[-1])

  save_obj(agent, os.path.join(agent_folder, 'agent_final'))
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_invspec_gp.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
