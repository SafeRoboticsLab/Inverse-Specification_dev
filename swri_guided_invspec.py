# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from typing import Any, Tuple, Union
import time
import os
import functools
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt
from queue import PriorityQueue

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblemInvSpec, SWRISimParallel

# human simulator module
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRankerSimulator

# inverse specification module
from funct_approx.config import GPConfig
from invspec.inv_spec import InvSpec
from invspec.query_selector.random_selector import RandomQuerySelector
from invspec.query_selector.upper_confidence_selector import UCBQuerySelector
from invspec.inference.reward_GP import RewardGP

# design optimization module
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)
from pymoo.core.population import Population

# others
from utils import (
    set_seed, save_obj, load_obj, query_and_collect, plot_result_pairwise,
    plot_single_objective
)
from config.config import load_config
from shutil import copyfile


def get_indices(
    agent: Any, pop: Union[Population, np.ndarray], num_query: int,
    query_select_type: str
) -> np.ndarray:
  if query_select_type == "random_pair":
    return agent.get_query(pop, num_query)
  elif query_select_type == "rand":
    return np.random.choice(len(pop), num_query)
  elif query_select_type == "ucb":
    return agent.get_query(
        pop, num_query, eval_func=agent.inference.get_ucb, add_cur_best=True
    )
  else:
    raise ValueError(
        "query_select_type({}) not supported".format(query_select_type)
    )


def get_design_from_heap(heap: PriorityQueue) -> Any:
  assert not heap.empty(), "The heap is empty!"
  if heap.qsize() == 1:
    idx_heap = 0
  else:
    idx_heap = np.random.choice(heap.qsize() - 1) + 1

  return heap.queue[idx_heap]


def idx2design(
    idx: int, heap: PriorityQueue, features: np.ndarray, components: np.ndarray
) -> Tuple[bool, np.ndarray, np.ndarray]:
  if idx == -1:
    old_design = get_design_from_heap(heap)
    return True, old_design.features, old_design.components
  else:
    return False, features[idx:idx + 1, :], components[idx:idx + 1, :]


def indices2query(
    indices: np.ndarray, heap: PriorityQueue, features: np.ndarray,
    components: np.ndarray
) -> Tuple[bool, np.ndarray, np.ndarray]:
  from_heap_0, feature_0, component_0 = idx2design(
      indices[0], heap, features, components
  )
  from_heap_1, feature_1, component_1 = idx2design(
      indices[1], heap, features, components
  )
  has_old = from_heap_0 or from_heap_1

  # if there is an old design, put it to the first of the query
  if from_heap_1:
    query_features = np.concatenate((feature_1, feature_0), axis=0)
    query_components = np.concatenate((component_1, component_0), axis=0)
  else:
    query_features = np.concatenate((feature_0, feature_1), axis=0)
    query_components = np.concatenate((component_0, component_1), axis=0)
  return has_old, query_features, query_components


def main(config_file: str, config_dict: dict) -> None:
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_ga = config_dict['GA']
  config_inv_spec = config_dict['INV_SPEC']
  config_gp = config_dict['GP']
  config_human = config_dict['HUMAN']

  out_folder = os.path.join('scratch', 'swri', 'guided')
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  os.makedirs(out_folder, exist_ok=True)
  fig_folder = os.path.join(out_folder, 'figure')
  os.makedirs(fig_folder, exist_ok=True)
  agent_folder = os.path.join(out_folder, 'agent')
  os.makedirs(agent_folder, exist_ok=True)
  obj_eval_folder = os.path.join(out_folder, 'obj_eval')
  os.makedirs(obj_eval_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))

  # endregion

  # region: == Define Problem ==
  print("\n== Problem ==")
  set_seed(seed_val=config_general.SEED, use_torch=False)
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblemInvSpec(
      TEMPLATE_FILE,
      EXEC_FILE,
      num_workers=config_general.NUM_WORKERS,
      inference=None,  #! a dummy init, needs to pass inference in later
      prefix="guided_" + time.strftime("%m-%d-%H_%M") + "_"
  )

  feature_names = problem.sim.objective_names
  print('features', feature_names)
  print('inputs:', problem.sim.input_names)

  features_bound = np.array([
      [0, 4000],
      [-400, 0],
      [0, 30],
      [-50, 0.],
      [-12, 0.],
  ])
  input_names_dict = {}
  for i in range(len(problem.sim.input_names)):
    input_names_dict['o' + str(i + 1)] = problem.input_names[i][8:]
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  scores_bound = np.array([-1e-8, 430])
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
  # endregion

  # region: == Define Inverse Specification ==
  print("\n== InvSpec Construction ==")
  CONFIG = GPConfig(
      SEED=config_general.SEED, HORIZONTAL_LENGTH=config_gp.HORIZONTAL_LENGTH,
      VERTICAL_VARIATION=config_gp.VERTICAL_VARIATION,
      NOISE_LEVEL=config_gp.NOISE_LEVEL, BETA=config_inv_spec.BETA,
      MEMORY_CAPACITY=config_inv_spec.MAX_QUERIES
  )
  print(vars(CONFIG), '\n')

  if config_inv_spec.POP_EXTRACT_TYPE == 'F':
    dimension = len(problem.sim.objective_names)
  elif config_inv_spec.POP_EXTRACT_TYPE == 'X':
    dimension = len(problem.sim.input_names)
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
      input_bound = features_bound
    elif config_inv_spec.POP_EXTRACT_TYPE == 'X':
      input_bound = component_values_bound
    input_min = input_bound[:, 0]
    input_max = input_bound[:, 1]
  else:
    input_normalize = False
    input_bound = None
    input_min = None
    input_max = None

  if config_inv_spec.QUERY_SELECTOR_TYPE == "ucb":
    query_selector = UCBQuerySelector(tradeoff=config_inv_spec.TRADEOFF)
  else:
    query_selector = RandomQuerySelector()

  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initial_point, input_min=input_min,
          input_max=input_max, input_normalize=input_normalize,
          pop_extract_type=config_inv_spec.POP_EXTRACT_TYPE, verbose=True
      ), query_selector=query_selector
  )
  # endregion

  # region: == Inverse Specification Inits ==
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
      #! hacky: just assume "not distinguishable (equal)" is "lower than" to
      #! prevent asking same query for two times
      return False

  designs_heap = PriorityQueue()
  num_query_per_batch = int(config_general.NUM_WORKERS / 2)
  init_obj_pop = load_obj(config_ga.INIT_OBJ_PATH)
  features = -init_obj_pop.get('F')  # we want to maximize
  components = init_obj_pop.get('X')

  for num_iter in range(config_inv_spec.MAX_ITER):
    n_ask = human.get_num_ranking_queries()
    if n_ask >= config_inv_spec.MAX_QUERIES_INIT:
      break
    print(num_iter, end=': ')

    # get sampled designs
    if designs_heap.empty():
      indices = get_indices(
          agent, init_obj_pop, num_query_per_batch, "random_pair"
      )
    else:
      indices = get_indices(
          agent, init_obj_pop, num_query_per_batch,
          config_inv_spec.QUERY_SELECTOR_TYPE
      )

    # interact with human
    for idx in indices:
      n_ask = human.get_num_ranking_queries()
      if n_ask >= config_inv_spec.MAX_QUERIES_INIT:
        break
      if designs_heap.empty():
        query_features = features[idx, :]
        query_components = components[idx, :]
        has_old = False
      else:
        if config_inv_spec.QUERY_SELECTOR_TYPE == "ucb":
          has_old, query_features, query_components = indices2query(
              idx, designs_heap, features, components
          )
        else:
          idx_tmp = np.array([-1, idx], dtype=int)
          has_old, query_features, query_components = indices2query(
              idx_tmp, designs_heap, features, components
          )
      valid, _ = query_and_collect(
          query_features, query_components, human, agent, config_inv_spec
      )

      add_to_heap = False
      if designs_heap.empty() and valid:  # both designs are new
        query_idx = 0 if valid == 1 else 1
        add_to_heap = True
      elif (not has_old) and valid:  # both designs are new
        query_idx = 0 if valid == 1 else 1
        add_to_heap = True
      elif valid == -1:
        query_idx = 1
        add_to_heap = True

      if add_to_heap:
        designs_heap.put(
            Design(
                query_features[query_idx:query_idx + 1, :],
                query_components[query_idx:query_idx + 1, :]
            )
        )
        print("Heap now has {} designs".format(designs_heap.qsize()))

    agent.learn()
    n_ask = human.get_num_ranking_queries()
    n_acc_fb = agent.get_number_feedback()
    print("Collect {:d} feedback out of {:d} queries".format(n_acc_fb, n_ask))

  # learn and save
  agent.learn()
  save_obj(agent, os.path.join(out_folder, 'agent', 'init'))
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

  algorithm = GA(
      pop_size=config_ga.POP_SIZE,
      n_offsprings=config_ga.POP_SIZE,
      sampling=sampling,
      crossover=crossover,
      mutation=mutation,
      eliminate_duplicates=True,
  )
  # endregion

  # region: == InvSpec-Guided GA ==
  print("\n== Optimization starts ==")
  obj = copy.deepcopy(algorithm)
  start_time = time.time()
  while obj.has_next():
    #= perform an iteration of the algorithm
    if not obj.is_initialized:
      # init obj, problem
      problem.update_inference(agent.inference)
      termination = get_termination("n_gen", config_ga.NUM_GEN)
      obj.setup(
          problem, termination=termination, seed=config_general.SEED,
          save_history=True
      )
      obj._initialize()
      init_obj_pop.set("n_gen", obj.n_gen)
      for individual in init_obj_pop:
        individual.evaluated = None
      obj.evaluator.eval(obj.problem, init_obj_pop)
      obj.advance(infills=init_obj_pop)
    else:
      obj.next()

    #= check performance
    if obj.n_gen % config_general.CHECK_GEN == 0:
      n_gen = obj.n_gen
      print(f"gen[{n_gen}]")

      # get the mesurements of the whole population
      component_values = obj.pop.get('X')
      features, oracle_scores, predicted_scores = problem.get_all(
          component_values
      )
      with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
        print(predicted_scores)
        print(oracle_scores)

      # plot oracle scores
      fig = plot_single_objective(
          oracle_scores, dict(o1="Score"), axis_bound=scores_bound
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(os.path.join(fig_folder, 'scores_' + str(n_gen) + '.png'))

      # plot features
      fig = plot_result_pairwise(
          len(feature_names), features, feature_names,
          axis_bound=features_bound
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(os.path.join(fig_folder, str(n_gen) + '.png'))

      # plot predicted scores
      fig = plot_single_objective(
          predicted_scores, dict(o1="Predicted Score"),
          axis_bound=np.array([-1e-8, 1.])
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(
          os.path.join(fig_folder, 'pred_scores_' + str(n_gen) + '.png')
      )

      # plot component values
      fig = plot_result_pairwise(
          len(problem.input_names), component_values, input_names_dict,
          axis_bound=component_values_bound, n_col_default=5, subfigsz=4,
          fsz=16, sz=20
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(os.path.join(fig_folder, 'inputs_' + str(n_gen) + '.png'))

      plt.close('all')

      res_dict = dict(
          features=features, component_values=component_values,
          scores=oracle_scores, predicted_scores=predicted_scores
      )
      obj_eval_path = os.path.join(obj_eval_folder, 'obj' + str(obj.n_gen))
      save_obj(res_dict, obj_eval_path)

    #= interact with human
    time2update = (obj.n_gen - 1) % config_inv_spec.INTERACT_PERIOD == 0
    if time2update and (obj.n_gen < config_ga.NUM_GEN) and (obj.n_gen > 1):
      print("\nAt generation {}".format(obj.n_gen))
      components = obj.pop.get('X')
      features, _, _ = problem.get_all(component_values)

      n_acc_fb = agent.get_number_feedback()
      n_want = config_inv_spec.MAX_QUERIES_PER
      if n_acc_fb + n_want > config_inv_spec.MAX_QUERIES:
        n_select = n_want - n_acc_fb
      else:
        n_select = n_want

      if n_select > 0:
        # 1. pick and send queries to humans
        # We need obj.pop since single-objective optimization only has one
        # optimum. Also, we passed in features/components directly since using
        # obj.pop.get("F") returns the predicted scores!
        if config_inv_spec.POP_EXTRACT_TYPE == "F":
          design = features
        else:
          design = components
        indices = get_indices(
            agent, design, n_select, config_inv_spec.QUERY_SELECTOR_TYPE
        )

        # 2. get feedback from humans
        for idx in indices:
          if config_inv_spec.QUERY_SELECTOR_TYPE == "ucb":
            has_old, query_features, query_components = indices2query(
                idx, designs_heap, features, components
            )
          else:
            idx_tmp = [-1, idx]
            has_old, query_features, query_components = indices2query(
                idx_tmp, designs_heap, features, components
            )

          valid, _ = query_and_collect(
              query_features, query_components, human, agent, config_inv_spec
          )
          add_to_heap = False
          if valid == -1:
            query_idx = 1
            add_to_heap = True
          elif (not has_old) and (valid == 1):
            query_idx = 0
            add_to_heap = True

          if add_to_heap:
            designs_heap.put(
                Design(
                    query_features[query_idx:query_idx + 1, :],
                    query_components[query_idx:query_idx + 1, :]
                )
            )
            print("Heap now has {} designs".format(designs_heap.qsize()))

        n_ask = human.get_num_ranking_queries()
        n_acc_fb = agent.get_number_feedback()
        print(
            "Collect {:d} feedback out of {:d} queries".format(
                n_acc_fb, n_ask
            )
        )

        # 3. update fitness function
        _ = agent.learn()
        problem.update_inference(agent.inference)
        obj.problem = copy.deepcopy(problem)
        for individual in obj.pop:
          individual.evaluated = None
        obj.evaluator.eval(obj.problem, obj.pop)  # re-evaluate

        print()
      else:
        print("Exceed maximal number of queries!", end=' ')
        print("Accumulated {:d} feedback".format(n_acc_fb))

  end_time = time.time()
  print("It took {:.1f} seconds".format(end_time - start_time))
  print("Save results ->", out_folder)
  components = obj.opt.get('X')
  features, oracle_scores, predicted_scores = problem.get_all(components)
  print(components)
  print(features)
  print(oracle_scores)
  print(predicted_scores)
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_guided_invspec_ucb.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
