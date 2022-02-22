# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import time
import os
import numpy as np
import argparse

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
from utils import set_seed, save_obj, normalize, unnormalize
from config.config import load_config
from shutil import copyfile


def sample_and_evaluate(problem, component_values_bound, num_samples=8):
  components = unnormalize(
      np.random.rand(num_samples, problem.n_var), component_values_bound[:, 0],
      component_values_bound[:, 1]
  )

  # get features
  y = {}
  problem._evaluate(components, y)
  return components, y


def query_and_collect(
    query_features, query_components, human, agent, config_inv_spec
):
  query = dict(F=query_features, X=query_components)
  # get feedback
  fb_raw = human.get_ranking(query)

  # store feedback
  if config_inv_spec.POP_EXTRACT_TYPE == 'F':
    inputs_to_invspec = query_features
  else:
    inputs_to_invspec = query_components

  if config_inv_spec.INPUT_NORMALIZE:
    inputs_to_invspec = agent.inference.normalize(inputs_to_invspec)
  q_1 = (inputs_to_invspec[0:1, :], np.array([]).reshape(1, 0))
  q_2 = (inputs_to_invspec[1:2, :], np.array([]).reshape(1, 0))

  if fb_raw != 2:
    if fb_raw == 0:
      fb_invspec = 1
    elif fb_raw == 1:
      fb_invspec = -1
    agent.store_feedback(q_1, q_2, fb_invspec)
    return True, inputs_to_invspec
  else:
    return False, None


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_inv_spec = config_dict['INV_SPEC']
  config_gp = config_dict['GP']
  config_human = config_dict['HUMAN']

  out_folder = os.path.join('scratch', 'swri', 'modular')
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config_invspec.yaml'))

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
  effective_query = []
  update_times = 0
  num_query_per_batch = int(config_general.NUM_WORKERS / 2)
  n_ask = 0

  # get the first valid query
  valid = False
  valid_query = None
  cnt = 0
  while valid_query is None:
    print(cnt, end=': ')
    # randomly sample component values
    components, y = sample_and_evaluate(
        problem, component_values_bound, config_general.NUM_WORKERS
    )
    features = y['F']
    oracle_scores_all = y['scores']

    for i in range(num_query_per_batch):
      start, end = 2 * i, 2*i + 2

      query_features = -features[start:end, :]
      query_components = components[start:end, :]
      oracle_scores = oracle_scores_all[start:end]

      valid, inputs_to_invspec = query_and_collect(
          query_features, query_components, human, agent, config_inv_spec
      )
      n_ask += 1
      if valid:
        effective_query.append([inputs_to_invspec, oracle_scores])
        valid_query = (query_features, query_components, oracle_scores)
        print("Get the first valid query!\n\n")
        break
    cnt += 1

  early_terminate = False
  # keep a valid query in buffer
  for num_iter in range(config_inv_spec.MAX_ITER):
    print(num_iter, end=': ')
    # randomly sample component values
    components, y = sample_and_evaluate(
        problem, component_values_bound, config_general.NUM_WORKERS
    )
    features = y['F']
    oracle_scores_all = y['scores']

    # compare the new designs with the old designs (valid query) in the buffer
    for i in range(config_general.NUM_WORKERS):
      new_feature = -features[i:i + 1, :]
      new_component = components[i:i + 1, :]
      new_oracle_score = oracle_scores_all[i:i + 1]

      valid_list = [False, False]
      candidate_query_list = [None, None]
      for j in range(2):
        old_feature = valid_query[0][j:j + 1, :]
        old_component = valid_query[1][j:j + 1, :]
        old_oracle_score = valid_query[2][j:j + 1, :]
        query_features = np.concatenate((old_feature, new_feature), axis=0)
        query_components = np.concatenate((old_component, new_component),
                                          axis=0)
        oracle_scores = np.concatenate((old_oracle_score, new_oracle_score))

        valid, inputs_to_invspec = query_and_collect(
            query_features, query_components, human, agent, config_inv_spec
        )
        candidate_query_list[j] = (
            query_features, query_components, oracle_scores
        )
        n_ask += 1
        if valid:
          effective_query.append([inputs_to_invspec, oracle_scores])
          valid_list[j] = True

      if valid_list[0]:
        if not valid_list[1]:
          valid_query = candidate_query_list[0]
        else:
          valid_query = candidate_query_list[np.random.choice(2)]
      elif valid_list[1]:
        valid_query = candidate_query_list[1]

    n_acc_fb = agent.get_number_feedback()
    print("Collect {:d} feedback out of {:d} queries".format(n_acc_fb, n_ask))
    if n_acc_fb >= config_inv_spec.REQUIRED_FB:
      break

    # update fitness function
    if (
        early_terminate
        and len(effective_query) >= config_inv_spec.NUM_REQUIRED_QUERY
    ):
      update_times += 1
      print("\nUpdate:")
      _ = agent.learn()

      # report
      normalized_oracle_scores = np.empty(shape=(2 * len(effective_query),))
      predicted_scores = np.empty(shape=(2 * len(effective_query),))
      for i, (inputs_to_invspec, oracle_scores) in enumerate(effective_query):
        start, end = 2 * i, 2*i + 2
        predicted_scores[start:end] = agent.inference.eval(inputs_to_invspec)
        normalized_oracle_scores[start:end] = normalize(
            oracle_scores.reshape(-1), scores_bound[0], scores_bound[1]
        )

      ratio = np.max(normalized_oracle_scores) / np.max(predicted_scores)
      scaled_predicted_scores = predicted_scores * ratio
      error = np.mean((normalized_oracle_scores - scaled_predicted_scores)**2)

      with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
        print(normalized_oracle_scores, scaled_predicted_scores, error)
      effective_query = []
      if error <= 1e-6:
        break

  n_acc_fb = agent.get_number_feedback()
  print("Collect {:d} feedback out of {:d} queries".format(n_acc_fb, n_ask))
  agent.learn()
  # endregion

  agent_path = os.path.join(out_folder, 'agent')
  save_obj(agent, agent_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_mod_invspec.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
