# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
# test #queries: python3 auv_InvSpec_GP.py -nw 25 -ng 100 -ip 25 -nq <number>
# default: python3 auv_InvSpec_GP.py -nw 0 -ng 300 -ip 25 -nq 2 -mq 20 -n def

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time
import os

os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from auv.problem import AUVsim
from utils import (
    set_seed, save_obj, get_infeasible_designs, get_inference_output
)
from utils import (
    plot_result_3D, plot_result_pairwise, plot_output_2D, plot_output_3D
)

# region: == ARGS ==
parser = argparse.ArgumentParser()

# problem parameters
parser.add_argument(
    "-p", "--problem_type", help="problem type", default='p2', type=str,
    choices=['p1', 'p2', 'p3']
)

# GA parameters
parser.add_argument(
    "-rnd", "--random_seed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-psz", "--pop_size", help="population size", default=25, type=int
)
parser.add_argument(
    "-nw", "--num_warmup", help="#warmup", default=25, type=int
)
parser.add_argument(
    "-ng", "--num_gen", help="#generation", default=100, type=int
)
parser.add_argument(
    "-st", "--survival_type", help="survival type", default='stoc', type=str,
    choices=['stoc', 'noisy_stoc', 'det', 'crowd']
)

# human simulator
parser.add_argument(
    "-b_h", "--beta_h", help="beta in the simulator", default=10, type=float
)
parser.add_argument(
    "-ht", "--human_type", help="human type", default='range_hard', type=str,
    choices=['speed', 'range', 'range_hard']
)
# parser.add_argument("-hw", "--humanWeights",    help="human weights",
#     default=[0.7, 0.1, 0.2],  nargs="*", type=float)

# human interface
parser.add_argument(
    "-qst", "--query_selector_type", help="query selector type", default='ig',
    type=str, choices=['rand', 'ig']
)
parser.add_argument(
    "-ip", "--interact_period", help="interaction period", default=25, type=int
)
parser.add_argument(
    "-nq", "--num_queries_per_update", help="#queries per update", default=1,
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
parser.add_argument("-n", "--name", help="extra name", default=None)

args = parser.parse_args()
print(args)
_outFolder = os.path.join(
    'scratch', 'AUV_' + args.problem_type,
    'InvSpec-GP-' + args.human_type + '-' + args.query_selector_type
)
if args.name is None:
  numQueries = int(np.ceil(float(args.num_gen) / args.interact_period))
  outFolder = os.path.join(_outFolder, str(numQueries) + 'queries')
else:
  outFolder = os.path.join(_outFolder, args.name)
print(outFolder)
figFolder = os.path.join(outFolder, 'figure')
os.makedirs(figFolder, exist_ok=True)
agentFolder = os.path.join(outFolder, 'agent')
os.makedirs(agentFolder, exist_ok=True)
objEvalFolder = os.path.join(outFolder, 'objEval')
os.makedirs(objEvalFolder, exist_ok=True)

# endregion


# region: == local functions ==
def getHumanScores(design_feature, w_opt, active_constraint_set=None):
  n_designs = design_feature.shape[0]
  indicator = get_infeasible_designs(design_feature, active_constraint_set)
  feasible_index = np.arange(n_designs)[np.logical_not(indicator)]
  feasible_designs = design_feature[feasible_index]
  scores = feasible_designs @ w_opt
  return feasible_index, scores


def report(
    agent, design_feature, w_opt, showRankNumber=5, active_constraint_set=None
):

  fitness = agent.inference.eval(design_feature)
  hard_order = np.argsort(-fitness)

  feasible_index, scores = getHumanScores(
      design_feature, w_opt, active_constraint_set=active_constraint_set
  )
  order = np.argsort(-scores)

  print("\nHard Thresholding:")
  print(hard_order[:showRankNumber])
  with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
    print(fitness[hard_order[:showRankNumber]])

  if order.shape[0] > 0:
    print("\nReal Order:")
    showRankNumber = min(order.shape[0], showRankNumber)
    print(feasible_index[order[:showRankNumber]])
    with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
      print(scores[order[:showRankNumber]])

    score_opt = w_opt @ design_feature[order[0]]
    score_pred = w_opt @ design_feature[hard_order[0]]
    score_diff = score_opt - score_pred
    print("Score difference: {:.3f}".format(score_diff))
  else:
    print("\nAll designs violate human internal preference!")


# endregion

# region: == Define Problem ==
set_seed(seed_val=args.random_seed, use_torch=True)
problem = AUVsim(problem_type=args.problem_type)
n_obj = problem.n_obj
objective_names = problem.fparams.func.objective_names
print(problem)
print('inputs:', problem.fparams.xinputs)

F_min = np.array([0., 0.4, -350.])
F_max = np.array([12000., 1.6, 0.])
axis_bound = np.empty(shape=(3, 2))
axis_bound[:, 0] = F_min
axis_bound[:, 1] = F_max
# endregion

# region: == Define Algorithm ==
from pymoo.factory import (
    get_termination, get_sampling, get_crossover, get_mutation
)
from invspec.nsga_invSpec import NSGA_INV_SPEC
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)
from pymoo.core.duplicate import DefaultDuplicateElimination
# from pymoo.configuration import Configuration

# Configuration.show_compile_hint = False

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

algorithm = NSGA_INV_SPEC(
    pop_size=args.pop_size, n_offsprings=args.pop_size, sampling=sampling,
    crossover=crossover, mutation=mutation,
    eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-3),
    warmup=args.num_warmup, beta=args.beta_m
)

# termination criterion
numGenTotal = args.num_gen
termination = get_termination("n_gen", numGenTotal)
# endregion

# region: == Define Human Simulator
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRanker

print("\n== Human Simulator ==")
active_constraint_set = None
if args.human_type == 'speed':
  w_opt = np.array([0.1, 0.7, 0.2])
else:
  w_opt = np.array([0.5, 0.1, 0.4])
  if args.human_type == 'range_hard':
    active_constraint_set = [['0', 0.2], ['1', 0.2]]
print("Human simulator has weights below")
print(w_opt)
if active_constraint_set is not None:
  print("Human simulator has active constraints:")
  print(active_constraint_set)

human = HumanSimulator(
    ranker=PairRanker(
        w_opt, beta=args.beta_h, active_constraint_set=active_constraint_set,
        perfect_rank=True
    )
)
# endregion

# region: == Define invSpec ==
from funct_approx.config import GPConfig

print("\n== InvSpec Construction ==")
CONFIG = GPConfig(
    SEED=args.random_seed, MAX_QUERIES=args.max_queries,
    MAX_QUERIES_PER=args.num_queries_per_update,
    HORIZONTAL_LENGTH=args.horizontal_length,
    VERTICAL_VARIATION=args.vertical_variation, NOISE_LEVEL=args.noise_level,
    NOISE_PROBIT=args.noise_probit
)
print(vars(CONFIG), '\n')

from invspec.inv_spec import InvSpec
from invspec.querySelector.random_selector import RandomQuerySelector
from invspec.querySelector.mutual_info_query_selector import (
    MutualInfoQuerySelector
)
from invspec.inference.reward_GP import RewardGP

dimension = w_opt.shape[0]
initialPoint = np.zeros(dimension)

if args.query_selector_type == 'rand':
  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initialPoint, F_min=F_min, F_max=F_max,
          verbose=True
      ), querySelector=RandomQuerySelector()
  )
else:
  agent = InvSpec(
      inference=RewardGP(
          dimension, 0, CONFIG, initialPoint, F_min=F_min, F_max=F_max,
          verbose=True
      ), querySelector=MutualInfoQuerySelector()
  )
# endregion

# region: == Optimization ==
print("\n== Optimization starts ...")
# perform a copy of the algorithm to ensure reproducibility
import copy

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
    F = -obj.opt.get('F')
    if n_obj == 3:
      fig = plot_result_3D(F, objective_names, axis_bound)
    else:
      fig = plot_result_pairwise(n_obj, F, objective_names, axis_bound)
    # fig.suptitle(args.survival_type, fontsize=20)
    fig.supxlabel(
        '{}-G{}: {} cumulative queries'.format(
            args.survival_type, n_gen, n_acc_fb
        ), fontsize=20
    )
    fig.tight_layout()
    figProFolder = os.path.join(figFolder, 'progress')
    os.makedirs(figProFolder, exist_ok=True)
    fig.savefig(os.path.join(figProFolder, str(n_gen) + '.png'))
    plt.close()

  #= interact with human
  if args.num_warmup == 0:
    time2update = ((obj.n_gen == 1) or (obj.n_gen % args.interact_period == 0))
  else:
    time2update = (obj.n_gen - args.num_warmup) % args.interact_period == 0

  if time2update and (obj.n_gen < numGenTotal):
    print("\nAt generation {}".format(obj.n_gen))
    F = agent.normalize(-obj.opt.get('F'))

    # if obj.n_gen == args.num_warmup and args.num_warmup != 0:
    #     n_want = 2
    # else:
    #     n_want = CONFIG.MAX_QUERIES_PER
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
        Ds = F[idx, :]
        fb = human.get_ranking(Ds)
        if obj.n_gen == args.num_warmup and args.num_warmup != 0:
          print(Ds, fb)

        q_1 = (Ds[0:1, :], action)
        q_2 = (Ds[1:2, :], action)

        if fb == 0:
          f = 1
        elif fb == 1:
          f = -1
        elif fb == 2:
          eps = np.random.uniform()
          f = 1 if eps > 0.5 else -1
        agent.store_feedback(q_1, q_2, f)
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
      indices = np.argsort(F[:, 0])
      F = F[indices]
      save_obj(agent, os.path.join(agentFolder, 'agent' + str(updateTimes)))
      objEvalPath = os.path.join(
          objEvalFolder, 'obj' + str(updateTimes - 1) + '.npy'
      )
      np.save(objEvalPath, F)
      feasible_index, scores = getHumanScores(F, w_opt, active_constraint_set)
      acc = len(feasible_index) / len(F)
      print('Feasible ratio: {:.3f}'.format(acc))
      with np.printoptions(formatter={'float': '{: .3f}'.format}):
        print(scores)
      print()
    else:
      print("Exceed maximal number of queries!", end=' ')
      print("Accumulated {:d} feedback".format(n_acc_fb))

    # report(agent, F, w_opt, showRankNumber=10,
    #     active_constraint_set=active_constraint_set)

end_time = time.time()
print("It took {:.1f} seconds".format(end_time - start_time))
# endregion

# region: finally obtain the result object
save_obj(agent, os.path.join(outFolder, 'agent'))

res = obj.result()
if args.use_timestr:
  timestr = time.strftime("%m-%d-%H_%M")
else:
  timestr = ''
picklePath = os.path.join(outFolder, timestr + 'res.pkl')
with open(picklePath, 'wb') as output:
  pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

F = -res.F
fig = plot_result_pairwise(
    n_obj, F, objective_names, axis_bound,
    active_constraint_set=active_constraint_set
)
fig.tight_layout()
fig.savefig(os.path.join(figFolder, 'objPairwise.png'))

indices = np.argsort(F[:, 0])
F = F[indices]
_F = agent.inference.normalize(F)
objEvalPath = os.path.join(objEvalFolder, 'obj' + str(updateTimes) + '.npy')
np.save(objEvalPath, _F)
feasible_index, scores = getHumanScores(_F, w_opt, active_constraint_set)
acc = len(feasible_index) / len(_F)
print()
print('Feasible ratio: {:.3f}'.format(acc))
with np.printoptions(formatter={'float': '{: .3f}'.format}):
  print(_F)
  print(scores)

#== plot output ==
obj_list_un = np.array([-150, -1])
obj_list = (obj_list_un
            - axis_bound[2, 0]) / (axis_bound[2, 1] - axis_bound[2, 0])
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
    X, Y, Z, obj_list_un, axis_bound, fsz=16, subfigsz=subfigsz, cm=cm
)
fig.tight_layout()
fig_path = os.path.join(
    figFolder, 'GP_op_' + args.query_selector_type + '.png'
)
fig.savefig(fig_path)

# 2D-plot
fig2, axes = plot_output_2D(
    X, Y, Z, obj_list_un, axis_bound, level_ratios, fsz=18, subfigsz=subfigsz,
    cm=cm, lw=lw
)
for ax in axes:
  # plot constraint threshold
  ax.plot([0.2, 0.2], [0, 1], 'r:', lw=lw - 0.5)
  ax.plot([0, 1], [0.2, 0.2], 'r:', lw=lw - 0.5)
fig2.tight_layout()
fig_path = os.path.join(
    figFolder, 'GP_op_2D_' + args.query_selector_type + '.png'
)
fig2.savefig(fig_path)
# endregion
