# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import copy
import time
import os
import numpy as np
import argparse
import functools
from queue import PriorityQueue

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblem, SWRISimParallel

# human simulator module
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRankerSimulator

# inverse specification module
from funct_approx.config import GPConfig
from invspec.inv_spec import InvSpec
from invspec.querySelector.random_selector import RandomQuerySelector
from invspec.inference.reward_GP import RewardGP

# others
from utils import (set_seed, save_obj, query_and_collect, sample_and_evaluate)
from config.config import load_config
from shutil import copyfile


def stop_asking(n_acc_fb, n_ask, config_inv_spec):
  if n_acc_fb >= config_inv_spec.REQUIRED_FB:
    return True
  if n_ask >= config_inv_spec.MAX_ASK:
    return True
  return False


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_inv_spec = config_dict['INV_SPEC']
  config_gp = config_dict['GP']
  config_human = config_dict['HUMAN']

  out_folder = os.path.join('scratch', 'swri', 'modular')
  early_folder = os.path.join(out_folder, config_general.EARLY_NAME)
  os.makedirs(early_folder, exist_ok=True)
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config_invspec.yaml'))
  copyfile(config_file, os.path.join(early_folder, 'config_invspec.yaml'))

  # endregion

  # region: == Define Problem ==
  print("\n== Problem ==")
  set_seed(seed_val=config_general.SEED, use_torch=False)
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblem(
      TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS,
      prefix="eval_" + time.strftime("%m-%d-%H_%M") + "_"
  )

  objective_names = problem.objective_names
  print('objectives', objective_names)
  print('inputs:', problem.input_names)

  objectives_bound = np.array([
      [0, 4000],
      [-400, 0],
      [0, 30],
      [-50, 0.],
      [-12, 0.],
  ])
  input_names_dict = {}
  for i in range(len(problem.input_names)):
    input_names_dict['o' + str(i + 1)] = problem.input_names[i][8:]
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  # endregion

  # region: == Define Human Simulator ==
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
  # endregion

  # region: == Define Inverse Specification ==
  print("\n== InvSpec Construction ==")
  CONFIG = GPConfig(
      SEED=config_general.SEED, HORIZONTAL_LENGTH=config_gp.HORIZONTAL_LENGTH,
      VERTICAL_VARIATION=config_gp.VERTICAL_VARIATION,
      NOISE_LEVEL=config_gp.NOISE_LEVEL, BETA=config_inv_spec.BETA,
      MEMORY_CAPACITY=config_inv_spec.REQUIRED_FB
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

  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initial_point, input_min=input_min,
          input_max=input_max, input_normalize=input_normalize,
          pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
      ), querySelector=RandomQuerySelector()
  )
  # endregion

  # region: == Inverse Specification Starts ==
  num_query_per_batch = int(config_general.NUM_WORKERS / 2)
  designs_heap = PriorityQueue()

  # Define class Design for heap
  @functools.total_ordering
  class Design:

    def __init__(self, features, components):
      self.features = features
      self.components = components

    def __repr__(self):
      return (
          "Design: features(" + repr(self.features) + "), components("
          + repr(self.components) + ")"
      )

    def __gt__(self, other):
      query_features = np.concatenate((self.features, other.features), axis=0)
      query_components = np.concatenate((self.components, other.components),
                                        axis=0)

      fb_invspec, _ = query_and_collect(
          query_features, query_components, human, agent, config_inv_spec,
          collect_undistinguished=False
      )
      return fb_invspec == 1

    def __eq__(self, other):
      #! hacky: just assume not distinguished (equal) one is lower than to
      #! prevent asking same query for two times
      return False

  # get the first valid query
  valid = False
  cnt = 0
  while designs_heap.empty():
    print(cnt, end=': ')
    # randomly sample component values
    components, y = sample_and_evaluate(
        problem, component_values_bound, config_general.NUM_WORKERS
    )
    features = y['F']

    for i in range(num_query_per_batch):
      start, end = 2 * i, 2*i + 2

      query_features = -features[start:end, :]
      query_components = components[start:end, :]

      valid, _ = query_and_collect(
          query_features, query_components, human, agent, config_inv_spec
      )
      if valid:
        print("Get the first valid query: {}!".format(valid), end="\n\n")
        if valid == 1:
          designs_heap.put(
              Design(query_features[0:1, :], query_components[0:1, :])
          )
        else:
          designs_heap.put(
              Design(query_features[1:2, :], query_components[1:2, :])
          )
        break
    cnt += 1

  # keep a heap
  stored_early = False
  for num_iter in range(config_inv_spec.MAX_ITER):
    print("\nAfter", num_iter, "main iterations", end=': ')
    n_acc_fb = agent.get_number_feedback()
    n_ask = human.get_num_ranking_queries()
    print("Collect {:d} feedback out of {:d} queries".format(n_acc_fb, n_ask))
    if stop_asking(n_acc_fb, n_ask, config_inv_spec):
      break
    # randomly sample component values
    components, y = sample_and_evaluate(
        problem, component_values_bound, config_general.NUM_WORKERS
    )
    features = y['F']

    # compare the new designs with the old designs in the heap
    for i in range(config_general.NUM_WORKERS):
      n_acc_fb = agent.get_number_feedback()
      n_ask = human.get_num_ranking_queries()
      if n_ask >= config_inv_spec.EARLY_STORE and not stored_early:
        agent.learn()
        stored_early = True
        save_obj(copy.deepcopy(agent), os.path.join(early_folder, 'agent'))
      if stop_asking(n_acc_fb, n_ask, config_inv_spec):
        break
      print("Testing", "new design", i, end=":")
      new_feature = -features[i:i + 1, :]
      new_component = components[i:i + 1, :]
      # select the worst effective design or a random design in the buffer
      if np.random.rand() > config_inv_spec.RANDOM_SELECT_RATE:
        old_feature = designs_heap.queue[0].features
        old_component = designs_heap.queue[0].components
      else:
        if designs_heap.qsize() == 1:
          index = 0
        else:
          index = np.random.choice(designs_heap.qsize() - 1) + 1
        old_feature = designs_heap.queue[index].features
        old_component = designs_heap.queue[index].components

      query_features = np.concatenate((old_feature, new_feature), axis=0)
      query_components = np.concatenate((old_component, new_component), axis=0)
      valid, _ = query_and_collect(
          query_features, query_components, human, agent, config_inv_spec
      )
      if valid == -1:  # new design is preferred
        designs_heap.put(Design(new_feature, new_component))
        print("Heap now has {} designs".format(designs_heap.qsize()))

  agent.learn()
  save_obj(agent, os.path.join(out_folder, 'agent'))
  components = []
  for design in designs_heap.queue:
    components.append(design.components.reshape(-1))
  y = {}
  components = np.array(components)
  problem._evaluate(components, y)
  print(y['scores'])
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_mod_invspec_heap.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
