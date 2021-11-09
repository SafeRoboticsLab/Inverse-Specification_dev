# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
# example: python3 swri_ga_scores.py -cg 1 -ng 50 -psz 10

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblem

# design optimization module
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)

# others
from utils import set_seed, plot_single_objective, plot_result_pairwise

timestr = time.strftime("%m-%d-%H_%M")

# region: == ARGS ==
parser = argparse.ArgumentParser()

# GA parameters
parser.add_argument(
    "-rnd", "--random_seed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-psz", "--pop_size", help="population size", default=25, type=int
)
parser.add_argument(
    "-ng", "--num_gen", help="#generation", default=200, type=int
)
parser.add_argument(
    "-cg", "--check_generation", help="check period", default=1, type=int
)
parser.add_argument("-n", "--name", help="extra name", default="")

args = parser.parse_args()
print(args)
out_folder = os.path.join('scratch', 'swri', 'GA_single' + args.name)
fig_folder = os.path.join(out_folder, 'figure')
os.makedirs(fig_folder, exist_ok=True)
# endregion

# region: == Define Problem ==
set_seed(seed_val=args.random_seed, use_torch=True)
TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
EXEC_FILE = os.path.join('swri', "new_fdm")
values_to_extract = np.array(["Path_traverse_score_based_on_requirements"])
objective_names = dict(o1="Score")
obj_indicator = np.array([-1.])
problem = SWRIProblem(
    TEMPLATE_FILE,
    EXEC_FILE,
    num_workers=5,
    values_to_extract=values_to_extract,
    objective_names=objective_names,
    obj_indicator=obj_indicator,
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
for key, value in y.items():
  print(key, ":", value)
# endregion

# region: == Define Algorithm ==
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
    pop_size=args.pop_size,
    n_offsprings=args.pop_size,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
)
# termination criterion
termination = get_termination("n_gen", args.num_gen)
# endregion

# region: == Define Solver ==
print("\n== Optimization starts ...")
# perform a copy of the algorithm to ensure reproducibility
obj = copy.deepcopy(algorithm)

# let the algorithm know what problem we are intending to solve and provide
# other attributes
obj.setup(
    problem, termination=termination, seed=args.random_seed, save_history=True
)
fig_progress_folder = os.path.join(fig_folder, 'progress')
os.makedirs(fig_progress_folder, exist_ok=True)

# until the termination criterion has not been met
start_time = time.time()
while obj.has_next():
  print(obj.n_gen, end='\r')
  # perform an iteration of the algorithm
  obj.next()

  # check performance
  if (obj.n_gen - 1) % args.check_generation == 0:
    n_gen = obj.n_gen
    n_nds = len(obj.opt)
    print(f"gen[{n_gen}]: n_nds: {n_nds}")

    # plot the whole population
    F = -obj.pop.get('F')
    fig = plot_single_objective(F, objective_names)
    fig.supxlabel(str(n_gen), fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_progress_folder, str(n_gen) + '.png'))
    plt.close()

# finally obtain the result object
res = obj.result()
print("--> EXEC TIME: {}".format(time.time() - start_time))
# endregion

# region: finally obtain the result object
res = obj.result()
res_to_save = dict(X=res.X, F=res.F, pop=res.pop, opt=res.opt)
# for key, value in res_to_save.items():
#   print(key, ":", value)
picklePath = os.path.join(out_folder, timestr + '.pkl')
with open(picklePath, 'wb') as output:
  pickle.dump(res_to_save, output, pickle.HIGHEST_PROTOCOL)
print(picklePath)

F = -obj.pop.get('F').reshape(-1)
X = obj.pop.get('X')
fig = plot_single_objective(F, objective_names)
fig.tight_layout()
fig.savefig(os.path.join(fig_folder, 'obj_pairwise.png'))

indices = np.argsort(F)
F = F[indices]
with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
  print(F)

input_names_dict = {}
for i in range(len(problem.input_names)):
  input_names_dict['o' + str(i + 1)] = problem.input_names[i]
fig = plot_result_pairwise(
    len(problem.input_names), X, input_names_dict, axis_bound=None,
    n_col_default=5, subfigsz=4, fsz=16, sz=20
)
fig.tight_layout()
fig.savefig(os.path.join(fig_folder, 'input_pairwise.png'))

# endregion
