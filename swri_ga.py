# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import time
import os
import copy
import numpy as np
import argparse
import pickle

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblem
from swri.utils import report_pop_swri

# design optimization module
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)

# others
from utils import set_seed, plot_result_pairwise, save_obj, load_obj
from config.config import load_config
from shutil import copyfile

timestr = time.strftime("%m-%d-%H_%M")


def main(config_file, config_dict):
  # region: == init ==
  config_general = config_dict['GENERAL']
  config_ga = config_dict['GA']

  out_folder = os.path.join('scratch', 'swri', 'NSGA2')
  if config_general.NAME is not None:
    out_folder = os.path.join(out_folder, config_general.NAME)
  fig_folder = os.path.join(out_folder, 'figure')
  os.makedirs(fig_folder, exist_ok=True)
  obj_eval_folder = os.path.join(out_folder, 'obj_eval')
  os.makedirs(obj_eval_folder, exist_ok=True)
  fig_progress_folder = os.path.join(fig_folder, 'progress')
  os.makedirs(fig_progress_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config_ga.yaml'))

  init_with_pop = False
  if config_general.INIT_OBJ_PATH:
    init_with_pop = True
  # endregion

  # region: == Define Problem ==
  print("\n== Problem ==")
  set_seed(seed_val=config_general.SEED, use_torch=False)
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblem(
      TEMPLATE_FILE, EXEC_FILE, num_workers=config_general.NUM_WORKERS
  )
  n_obj = problem.n_obj
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

  algorithm = NSGA2(
      pop_size=config_ga.POP_SIZE,
      n_offsprings=config_ga.POP_SIZE,
      sampling=sampling,
      crossover=crossover,
      mutation=mutation,
      eliminate_duplicates=True,
  )

  # termination criterion
  termination = get_termination("n_gen", config_ga.NUM_GEN)
  # endregion

  # region: == Define Solver ==
  print("\n== Optimization starts ...")
  # perform a copy of the algorithm to ensure reproducibility
  obj = copy.deepcopy(algorithm)

  # let the algorithm know what problem we are intending to solve and provide
  # other attributes
  obj.setup(
      problem, termination=termination, seed=config_general.SEED,
      save_history=True
  )

  # if without initialiation, it goes through
  # 1. pymoo.core.algorithm.Algorithm.next()
  # 2. pymoo.core.algorithm.Algorithm.infill()
  #     a. pymoo.core.algorithm.Algorithm._initialize()
  #     b. pymoo.algorithms.base.genetic.GeneticAlgorithm._initialize_infill()
  # 3. pymoo.core.algorithm.Algorithm.advance(infills)
  #     a. pymoo.algorithms.base.genetic.GeneticAlgorithm._initialize_advance()
  if init_with_pop:
    init_obj_pop = load_obj(config_general.INIT_OBJ_PATH)
    obj._initialize()
    init_obj_pop.set("n_gen", obj.n_gen)
    obj.evaluator.eval(obj.problem, init_obj_pop, algorithm=obj)
    obj.advance(infills=init_obj_pop)

    features, component_values, scores = report_pop_swri(
        obj, fig_progress_folder, 0, objective_names, input_names_dict,
        objectives_bound, scores_bound, component_values_bound
    )

    res_dict = dict(
        features=features, component_values=component_values, scores=scores
    )
    obj_eval_path = os.path.join(obj_eval_folder, 'obj' + str(obj.n_gen))
    save_obj(res_dict, obj_eval_path)

  # until the termination criterion has not been met
  start_time = time.time()
  while obj.has_next():
    print(obj.n_gen, end='\r')
    # perform an iteration of the algorithm
    obj.next()

    if obj.n_gen == 20 and not init_with_pop:
      save_obj(obj.pop, os.path.join(out_folder, 'objective_20'))

    # check performance
    if obj.n_gen % config_general.CHECK_GEN == 0:
      features, component_values, scores = report_pop_swri(
          obj, fig_progress_folder, 0, objective_names, input_names_dict,
          objectives_bound, scores_bound, component_values_bound
      )
      res_dict = dict(
          features=features, component_values=component_values, scores=scores
      )
      obj_eval_path = os.path.join(obj_eval_folder, 'obj' + str(obj.n_gen))
      save_obj(res_dict, obj_eval_path)

  # finally obtain the result object
  res = obj.result()
  print("--> EXEC TIME: {}".format(time.time() - start_time))
  # endregion

  # region: finally obtain the result object
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
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "swri_nsga2.yaml")
  )
  args = parser.parse_args()
  config_dict = load_config(args.config_file)
  main(args.config_file, config_dict)
