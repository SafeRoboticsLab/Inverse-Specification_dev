# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
# example: python3 swri_invspec_gp.py -cg 5 -psz 10 -ng 50 -n def -nq 5

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from swri.problem import SWRIProblem, SWRISimParallel

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
from utils import (
    set_seed, save_obj, plot_result_3D, plot_result_pairwise,
    plot_single_objective, plot_output_2D, plot_output_3D
)

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
    "-nw", "--num_warmup", help="#warmup", default=10, type=int
)
parser.add_argument(
    "-ng", "--num_gen", help="#generation", default=200, type=int
)

# human simulator
parser.add_argument(
    "-b_h", "--beta_h", help="beta in the simulator", default=10, type=float
)
parser.add_argument(
    "-ht", "--human_type", help="human type", type=str, default="plain",
    choices=["plain", "has_const"]
)

# human interface
parser.add_argument(
    "-qst", "--query_selector_type", help="query selector type", default='ig',
    type=str, choices=['rand', 'ig']
)
parser.add_argument(
    "-ip", "--interact_period", help="interaction period", default=10, type=int
)
parser.add_argument(
    "-nq", "--num_queries_per_update", help="#queries per update", default=5,
    type=int
)
parser.add_argument(
    "-mq", "--max_queries", help="maximal #queries", default=50, type=int
)

# GP model
parser.add_argument(
    "-b_m", "--beta_m", help="beta in the model", default=10, type=float
)
parser.add_argument(
    "-hl", "--horizontal_length", help="ell in RBF kernel", default=.3,
    type=float
)
parser.add_argument(
    "-vv", "--vertical_variation", help="sigma_f in RBF kernel", default=1.,
    type=float
)
parser.add_argument(
    "-nl", "--noise_level", help="sigma_n in RBF kernel", default=0.1,
    type=float
)
parser.add_argument(
    "-np", "--noise_probit", help="sigma in Probit model", default=0.5,
    type=float
)

# output
parser.add_argument(
    "-cg", "--check_generation", help="check generation", default=1, type=int
)
parser.add_argument(
    "-uts", "--use_timestr", help="use timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default="")

args = parser.parse_args()
print(args)
out_folder = os.path.join('scratch', 'swri', 'InvSpec_GP', args.name)
fig_folder = os.path.join(out_folder, 'figure')
os.makedirs(fig_folder, exist_ok=True)
agent_folder = os.path.join(out_folder, 'agent')
os.makedirs(agent_folder, exist_ok=True)
obj_eval_folder = os.path.join(out_folder, 'obj_eval')
os.makedirs(obj_eval_folder, exist_ok=True)
# endregion

# region: == Define Problem ==
print("\n== Problem ==")
set_seed(seed_val=args.random_seed, use_torch=False)
TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
EXEC_FILE = os.path.join('swri', "new_fdm")
values_to_extract = np.array(["Path_traverse_score_based_on_requirements"])
problem = SWRIProblem(TEMPLATE_FILE, EXEC_FILE, num_workers=5)

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
problem._evaluate(x, y, get_score=True)
print(y['F'])
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

algorithm = NSGAInvSpec(
    pop_size=args.pop_size, n_offsprings=args.pop_size, sampling=sampling,
    crossover=crossover, mutation=mutation,
    eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-3),
    warmup=args.num_warmup, beta=args.beta_m
)

# termination criterion
numGenTotal = args.num_gen
termination = get_termination("n_gen", numGenTotal)
# endregion

# region: == Define Human Simulator ==
print("\n== Human Simulator ==")
active_constraint_set = None
if args.human_type == 'has_const':
  active_constraint_set = [['0', 0.2], ['1', 0.2]]
if active_constraint_set is not None:
  print("Human simulator has active constraints:")
  print(active_constraint_set)

human = HumanSimulator(
    ranker=PairRankerSimulator(
        simulator=SWRISimParallel(
            TEMPLATE_FILE, EXEC_FILE, num_workers=5, prefix='human_'
        ),
        beta=args.beta_h,
        active_constraint_set=active_constraint_set,
        perfect_rank=True,
    )
)
print(human.ranker._get_scores(query=dict(X=x), feasible_index=[0]))
# endregion

# region: == Define invSpec ==
print("\n== InvSpec Construction ==")
CONFIG = GPConfig(
    SEED=args.random_seed, MAX_QUERIES=args.max_queries,
    MAX_QUERIES_PER=args.num_queries_per_update,
    HORIZONTAL_LENGTH=args.horizontal_length,
    VERTICAL_VARIATION=args.vertical_variation, NOISE_LEVEL=args.noise_level,
    NOISE_PROBIT=args.noise_probit
)
print(vars(CONFIG), '\n')

dimension = problem.n_obj
initialPoint = np.zeros(dimension)

if args.query_selector_type == 'rand':
  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initialPoint, F_normalize=False, verbose=True
      ), querySelector=RandomQuerySelector()
  )
else:
  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initialPoint, F_normalize=False, verbose=True
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
    problem, termination=termination, seed=args.random_seed, save_history=True
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
  if obj.n_gen % args.check_generation == 0:
    n_gen = obj.n_gen
    n_nds = len(obj.opt)
    CV = obj.opt.get('CV').min()
    print(f"gen[{n_gen}]: n_nds: {n_nds} CV: {CV}")
    features = -obj.opt.get('F')
    if n_obj == 3:
      fig = plot_result_3D(features, objective_names, axis_bound=None)
    else:
      fig = plot_result_pairwise(
          n_obj, features, objective_names, axis_bound=None
      )
    fig.supxlabel(
        'G{}: {} cumulative queries'.format(n_gen, n_acc_fb), fontsize=20
    )
    fig.tight_layout()
    fig_progress_folder = os.path.join(fig_folder, 'progress')
    os.makedirs(fig_progress_folder, exist_ok=True)
    fig.savefig(os.path.join(fig_progress_folder, str(n_gen) + '.png'))

    X = obj.opt.get('X')
    out = {}
    problem._evaluate(X, out, get_score=True)
    print(out["F"].reshape(-1))
    fig = plot_single_objective(out["F"].reshape(-1), dict(o1="Score"))
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_progress_folder, 'score_' + str(n_gen) + '.png')
    )
    plt.close('all')

  #= interact with human
  if args.num_warmup == 0:
    time2update = ((obj.n_gen == 1) or (obj.n_gen % args.interact_period == 0))
  else:
    time2update = (obj.n_gen - args.num_warmup) % args.interact_period == 0

  if time2update and (obj.n_gen < numGenTotal):
    print("\nAt generation {}".format(obj.n_gen))
    features = -obj.pop.get('F')  # we want to maximize
    components = obj.pop.get('X')

    n_want = CONFIG.MAX_QUERIES_PER
    if n_acc_fb + n_want > CONFIG.MAX_QUERIES:
      n_ask = n_want - n_acc_fb
    else:
      n_ask = n_want

    if n_ask > 0:
      # 1. pick and send queries to humans
      indices = agent.get_query(obj.opt, n_ask)

      # 2. get feedback from humans
      action = np.array([]).reshape(1, 0)
      for idx in indices:
        query_features = features[idx, :]
        query_components = components[idx, :]

        query = dict(F=query_features, X=query_components)
        feedback = human.get_ranking(query)
        print(feedback)

        q_1 = (query_features[0:1, :], action)
        q_2 = (query_features[1:2, :], action)

        if feedback == 0:
          fb_invspec = 1
        elif feedback == 1:
          fb_invspec = -1
        elif feedback == 2:
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
res = obj.result()
res_to_save = dict(X=res.X, F=res.F, pop=res.pop, opt=res.opt)
picklePath = os.path.join(out_folder, timestr + '.pkl')
with open(picklePath, 'wb') as output:
  pickle.dump(res_to_save, output, pickle.HIGHEST_PROTOCOL)
print(picklePath)

features = -res.F
fig = plot_result_pairwise(
    n_obj, features, objective_names, axis_bound=None, n_col_default=5,
    subfigsz=4, fsz=16, sz=20
)
fig.tight_layout()
fig.savefig(os.path.join(fig_folder, 'obj_pairwise.png'))

print("\npick the design in the optimal front that has maximal objective 1.")
indices = np.argsort(features[:, 0])
features = features[indices]
with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
  print(features[-1])

print()
X = res.X
out = {}
problem._evaluate(X, out, get_score=True)
print(out["F"].reshape(-1))
# endregion
