# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblemInvSpec

# design optimization module
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)

# others
from utils import (
    set_seed, save_obj, load_obj, plot_result_pairwise, plot_single_objective
)
from config.config import load_config
from shutil import copyfile


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_ga = config_dict['GA']

  out_folder = os.path.join('scratch', 'swri', 'modular')
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  fig_folder = os.path.join(out_folder, 'figure')
  os.makedirs(fig_folder, exist_ok=True)
  obj_eval_folder = os.path.join(out_folder, 'obj_eval')
  os.makedirs(obj_eval_folder, exist_ok=True)
  fig_progress_folder = os.path.join(fig_folder, 'progress')
  os.makedirs(fig_progress_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config_ga.yaml'))
  # endregion

  # region: == Define Problem ==
  print("\n== Problem ==")
  set_seed(seed_val=config_general.SEED, use_torch=False)
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")

  agent = load_obj(config_general.AGENT_PATH)
  copyfile(
      config_general.AGENT_PATH + ".pkl",
      os.path.join(out_folder, 'agent.pkl')
  )

  problem = SWRIProblemInvSpec(
      TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS,
      inference=copy.deepcopy(agent.inference),
      prefix="eval_" + time.strftime("%m-%d-%H_%M") + "_"
  )
  feature_names = dict(
      o1="Distance",
      o2="Time",
      o3="Speed_avg",
      o4="Dist_err_max",
      o5="Dist_err_avg",
  )
  feature_bound = np.array([
      [0, 4000],
      [-400, 0],
      [0, 30],
      [-50, 0.],
      [-12, 0.],
  ])

  objective_names = problem.objective_names
  print('objectives', objective_names)
  print('inputs:', problem.input_names)

  x = np.array([[3.99, 3.67, 3.35, 3.03, 4.42, 17.],
                [3.99, 3.67, 3.35, 3.03, 4.42, 17.]])
  features, oracle_scores, predicted_scores = problem.get_all(x)
  print("\nGet ax example output from the problem:")
  print(features)
  print(oracle_scores)
  print(predicted_scores)

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

  algorithm = GA(
      pop_size=config_ga.POP_SIZE,
      n_offsprings=config_ga.POP_SIZE,
      sampling=sampling,
      crossover=crossover,
      mutation=mutation,
      eliminate_duplicates=True,
  )

  # termination criterion
  num_gen_total = config_ga.NUM_GEN
  termination = get_termination("n_gen", num_gen_total)
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
  start_time = time.time()
  while obj.has_next():
    print(obj.n_gen, end='\r')
    # perform an iteration of the algorithm
    obj.next()

    # check performance
    if obj.n_gen % config_general.CHECK_GEN == 0:
      n_gen = obj.n_gen
      n_nds = len(obj.opt)
      print(f"gen[{n_gen}]: n_nds: {n_nds}")

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
      fig.savefig(
          os.path.join(fig_progress_folder, 'scores_' + str(n_gen) + '.png')
      )

      # plot features
      fig = plot_result_pairwise(
          len(feature_names), features, feature_names, axis_bound=feature_bound
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(os.path.join(fig_progress_folder, str(n_gen) + '.png'))

      # plot predicted scores
      fig = plot_single_objective(
          predicted_scores, dict(o1="Predicted Score"),
          axis_bound=np.array([-1e-8, 1.])
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(
          os.path.join(
              fig_progress_folder, 'pred_scores_' + str(n_gen) + '.png'
          )
      )

      # plot component values
      fig = plot_result_pairwise(
          len(problem.input_names), component_values, input_names_dict,
          axis_bound=component_values_bound, n_col_default=5, subfigsz=4,
          fsz=16, sz=20
      )
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      fig.savefig(
          os.path.join(fig_progress_folder, 'inputs_' + str(n_gen) + '.png')
      )

      plt.close('all')

      res_dict = dict(
          features=features, component_values=component_values,
          scores=oracle_scores, predicted_scores=predicted_scores
      )
      obj_eval_path = os.path.join(obj_eval_folder, 'obj' + str(obj.n_gen))
      save_obj(res_dict, obj_eval_path)

  end_time = time.time()
  print("It took {:.1f} seconds".format(end_time - start_time))
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_mod_ga_heap.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
