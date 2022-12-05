# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from __future__ import annotations
import copy
import time
import os
import numpy as np
import argparse
from queue import PriorityQueue

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblem, SWRISimParallel

# human simulator module
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRankerSimulator

# inverse specification module
from funct_approx.config import GPConfig
from invspec.inv_spec import InvSpec
from invspec.query_selector.random_selector import RandomQuerySelector
from invspec.inference.reward_GP import RewardGP
from invspec.design import Design

# others
from utils import (
    set_seed, save_obj, query_and_collect, sample_and_evaluate, CompareDesign,
    get_random_design_from_heap
)
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
  query_key = "test" + str(problem.test_path)
  # endregion

  # region: == Define Human Simulator ==
  active_constraint_set = None
  if config_human.TYPE == 'has_const':
    active_constraint_set = [('0', 0.2), ('1', 0.2)]
  if active_constraint_set is not None:
    print("Human simulator has active constraints:")
    print(active_constraint_set)

  human = HumanSimulator(
      ranker=PairRankerSimulator(
          simulator=SWRISimParallel(
              TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS,
              prefix='human_'
          ),
          beta=config_human.BETA,
          active_constraint_set=active_constraint_set,
          perfect_rank=config_human.PERFECT_RANK,
          indifference=config_human.INDIFFERENCE,
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

  dimension = problem.n_obj
  # dimension = problem.n_var

  initial_point = np.zeros(dimension)
  if config_inv_spec.INPUT_NORMALIZE:
    input_normalize = True
    input_bound = objectives_bound
    # input_bound = component_values_bound
    input_min = input_bound[:, 0]
    input_max = input_bound[:, 1]
  else:
    input_normalize = False
    input_bound = None
    input_min = None
    input_max = None

  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, query_key, initial_point, input_min=input_min,
          input_max=input_max, input_normalize=input_normalize, verbose=True
      ),
      query_selector=RandomQuerySelector(),
  )
  # endregion

  # region: == Inverse Specification Starts ==
  num_query_per_batch = int(config_general.NUM_WORKERS / 2)
  designs_heap = PriorityQueue()
  CompareDesign.query_key = query_key
  CompareDesign.human = human
  CompareDesign.agent = agent
  CompareDesign.config = config_inv_spec

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

    designs = []
    for i in range(config_general.NUM_WORKERS):
      designs.append(
          Design(
              design_id=tuple(components[i, :]),
              physical_components={},
              test_params={query_key: components[i, :]},
              test_results={
                  query_key:
                      dict(
                          metrics=-features[i, :],
                          trajectory=np.empty(shape=(0,))
                      )
              },
              global_features={},
          )
      )

    for i in range(num_query_per_batch):
      query = designs[2 * i:2*i + 2]

      valid, _ = query_and_collect(
          query, query_key, human, agent, config_inv_spec
      )
      if valid:
        print("Get the first valid query: {}!".format(valid), end="\n\n")
        if valid == 1:
          designs_heap.put(CompareDesign(query[0]))
        else:
          designs_heap.put(CompareDesign(query[1]))
        break
    cnt += 1

  # Keeps a heap.
  stored_early = False
  for num_iter in range(config_inv_spec.MAX_ITER):
    print("\nAfter", num_iter, "main iterations", end=': ')
    n_acc_fb = agent.get_number_feedback()
    n_ask = human.get_num_ranking_queries()
    print("Collect {:d} feedback out of {:d} queries".format(n_acc_fb, n_ask))
    if stop_asking(n_acc_fb, n_ask, config_inv_spec):
      break
    # Samples component values randomly.
    components, y = sample_and_evaluate(
        problem, component_values_bound, config_general.NUM_WORKERS
    )
    features = y['F']

    # Compares the new designs with the old designs in the heap.
    for i in range(config_general.NUM_WORKERS):
      n_acc_fb = agent.get_number_feedback()
      n_ask = human.get_num_ranking_queries()
      # Stores inferred fitness with less queries for comparison.
      if n_ask >= config_inv_spec.EARLY_STORE and not stored_early:
        agent.learn()
        stored_early = True
        save_obj(copy.deepcopy(agent), os.path.join(early_folder, 'agent'))
      if stop_asking(n_acc_fb, n_ask, config_inv_spec):  # Early terminates.
        break
      print("Testing", "new design", i, end=":")
      new_design = Design(
          design_id=tuple(components[i, :]),
          physical_components={},
          test_params={query_key: components[i, :]},
          test_results={
              query_key:
                  dict(
                      metrics=-features[i, :], trajectory=np.empty(shape=(0,))
                  )
          },
          global_features={},
      )
      # Selects the worst effective design or a random design in the buffer
      if np.random.rand() > config_inv_spec.RANDOM_SELECT_RATE:
        old_design = designs_heap.queue[0].design
      else:
        old_design = get_random_design_from_heap(designs_heap).design

      query = [old_design, new_design]
      valid, _ = query_and_collect(
          query, query_key, human, agent, config_inv_spec
      )
      if valid == -1:  # new design is preferred
        designs_heap.put(CompareDesign(query[1]))
        print("Heap now has {} designs".format(designs_heap.qsize()))

  agent.learn()
  save_obj(agent, os.path.join(out_folder, 'agent'))
  components = np.array([
      design.design.get_test_params(query_key) for design in designs_heap.queue
  ], dtype=np.float32)
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
