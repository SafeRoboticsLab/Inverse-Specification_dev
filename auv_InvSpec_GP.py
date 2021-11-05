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
from utils import setSeed, save_obj, findInfeasibleDesigns, getInferenceOutput
from utils import plotResult3D, plotResultPairwise, plotOutput2D, plotOutput3D

# region: == ARGS ==
parser = argparse.ArgumentParser()

# problem parameters
parser.add_argument(
    "-p", "--problemType", help="problem type", default='p2', type=str,
    choices=['p1', 'p2', 'p3']
)

# GA parameters
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-psz", "--popSize", help="population size", default=25, type=int
)
parser.add_argument("-nw", "--numWarmup", help="#warmup", default=25, type=int)
parser.add_argument(
    "-ng", "--numGen", help="#generation", default=100, type=int
)
parser.add_argument(
    "-st", "--survivalType", help="survival type", default='stoc', type=str,
    choices=['stoc', 'noisy_stoc', 'det', 'crowd']
)

# human simulator
parser.add_argument(
    "-b_h", "--beta_h", help="beta in the simulator", default=10, type=float
)
parser.add_argument(
    "-ht", "--humanType", help="human type", default='range_hard', type=str,
    choices=['speed', 'range', 'range_hard']
)
# parser.add_argument("-hw", "--humanWeights",    help="human weights",
#     default=[0.7, 0.1, 0.2],  nargs="*", type=float)

# human interface
parser.add_argument(
    "-qst", "--querySelectorType", help="query selector type", default='ig',
    type=str, choices=['rand', 'ig']
)
parser.add_argument(
    "-ip", "--interactPeriod", help="interaction period", default=25, type=int
)
parser.add_argument(
    "-nq", "--numQueriesPer", help="#queries per update", default=1, type=int
)
parser.add_argument(
    "-mq", "--maxQueries", help="maximal #queries", default=50, type=int
)

# GP model
parser.add_argument(
    "-b_m", "--beta_m", help="beta in the model", default=10, type=float
)
parser.add_argument(
    "-hl", "--horizontalLength", help="ell in RBF kernel", default=.3,
    type=float
)
parser.add_argument(
    "-vv", "--verticalVariation", help="sigma_f in RBF kernel", default=1.,
    type=float
)
parser.add_argument(
    "-nl", "--noiseLevel", help="sigma_n in RBF kernel", default=0.1,
    type=float
)
parser.add_argument(
    "-np", "--noiseProbit", help="sigma in Probit model", default=0.5,
    type=float
)

# output
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=100, type=int
)
parser.add_argument(
    "-cg", "--checkGeneration", help="check generation", default=1, type=int
)
parser.add_argument(
    "-uts", "--useTimeStr", help="use timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default=None)

args = parser.parse_args()
print(args)
_outFolder = os.path.join(
    'scratch', 'AUV_' + args.problemType,
    'InvSpec-GP-' + args.humanType + '-' + args.querySelectorType
)
if args.name is None:
  numQueries = int(np.ceil(float(args.numGen) / args.interactPeriod))
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
def getHumanScores(designFeature, w_opt, activeConstraintSet=None):
  n_designs = designFeature.shape[0]
  indicator = findInfeasibleDesigns(designFeature, activeConstraintSet)
  feasibleIndex = np.arange(n_designs)[np.logical_not(indicator)]
  feasibleDesigns = designFeature[feasibleIndex]
  scores = feasibleDesigns @ w_opt
  return feasibleIndex, scores


def report(
    agent, designFeature, w_opt, showRankNumber=5, activeConstraintSet=None
):

  fitness = agent.inference.eval(designFeature)
  hard_order = np.argsort(-fitness)

  feasibleIndex, scores = getHumanScores(
      designFeature, w_opt, activeConstraintSet=activeConstraintSet
  )
  order = np.argsort(-scores)

  print("\nHard Thresholding:")
  print(hard_order[:showRankNumber])
  with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
    print(fitness[hard_order[:showRankNumber]])

  if order.shape[0] > 0:
    print("\nReal Order:")
    showRankNumber = min(order.shape[0], showRankNumber)
    print(feasibleIndex[order[:showRankNumber]])
    with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
      print(scores[order[:showRankNumber]])

    score_opt = w_opt @ designFeature[order[0]]
    score_pred = w_opt @ designFeature[hard_order[0]]
    score_diff = score_opt - score_pred
    print("Score difference: {:.3f}".format(score_diff))
  else:
    print("\nAll designs violate human internal preference!")


# endregion

# region: == Define Problem ==
setSeed(seed_val=args.randomSeed, useTorch=True)
problem = AUVsim(problemType=args.problemType)
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
    pop_size=args.popSize, n_offsprings=args.popSize, sampling=sampling,
    crossover=crossover, mutation=mutation,
    eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-3),
    warmup=args.numWarmup, beta=args.beta_m
)

# termination criterion
numGenTotal = args.numGen
termination = get_termination("n_gen", numGenTotal)
# endregion

# region: == Define Human Simulator
from humansim.human_simulator import HumanSimulator
from humansim.ranker.pair_ranker import PairRanker


print("\n== Human Simulator ==")
activeConstraintSet = None
if args.humanType == 'speed':
  w_opt = np.array([0.1, 0.7, 0.2])
else:
  w_opt = np.array([0.5, 0.1, 0.4])
  if args.humanType == 'range_hard':
    activeConstraintSet = [['0', 0.2], ['1', 0.2]]
print("Human simulator has weights below")
print(w_opt)
if activeConstraintSet is not None:
  print("Human simulator has active constraints:")
  print(activeConstraintSet)

human = HumanSimulator(
    ranker=PairRanker(
        w_opt, beta=args.beta_h, activeConstraintSet=activeConstraintSet,
        perfectRank=True
    )
)
# endregion

# region: == Define invSpec ==
from funct_approx.config import GPConfig


print("\n== InvSpec Construction ==")
CONFIG = GPConfig(
    SEED=args.randomSeed, MAX_QUERIES=args.maxQueries,
    MAX_QUERIES_PER=args.numQueriesPer,
    HORIZONTAL_LENGTH=args.horizontalLength,
    VERTICAL_VARIATION=args.verticalVariation, NOISE_LEVEL=args.noiseLevel,
    NOISE_PROBIT=args.noiseProbit
)
print(vars(CONFIG), '\n')

from invspec.inv_spec import InvSpec
from invspec.querySelector.random_selector import RandomQuerySelector
from invspec.querySelector.mutual_info_query_selector import (
    MutualInfoQuerySelector
)
from src.invspec.inference.reward_GP import RewardGP


dimension = w_opt.shape[0]
initialPoint = np.zeros(dimension)

if args.querySelectorType == 'rand':
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
    problem, termination=termination, seed=args.randomSeed, save_history=True
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
  if obj.n_gen % args.checkGeneration == 0:
    n_gen = obj.n_gen
    n_nds = len(obj.opt)
    CV = obj.opt.get('CV').min()
    print(f"gen[{n_gen}]: n_nds: {n_nds} CV: {CV}")
    F = -obj.opt.get('F')
    if n_obj == 3:
      fig = plotResult3D(F, objective_names, axis_bound)
    else:
      fig = plotResultPairwise(n_obj, F, objective_names, axis_bound)
    # fig.suptitle(args.survivalType, fontsize=20)
    fig.supxlabel(
        '{}-G{}: {} cumulative queries'.format(
            args.survivalType, n_gen, n_acc_fb
        ), fontsize=20
    )
    fig.tight_layout()
    figProFolder = os.path.join(figFolder, 'progress')
    os.makedirs(figProFolder, exist_ok=True)
    fig.savefig(os.path.join(figProFolder, str(n_gen) + '.png'))
    plt.close()

  #= interact with human
  if args.numWarmup == 0:
    time2update = ((obj.n_gen == 1) or (obj.n_gen % args.interactPeriod == 0))
  else:
    time2update = (obj.n_gen - args.numWarmup) % args.interactPeriod == 0

  if time2update and (obj.n_gen < numGenTotal):
    print("\nAt generation {}".format(obj.n_gen))
    F = agent.normalize(-obj.opt.get('F'))

    # if obj.n_gen == args.numWarmup and args.numWarmup != 0:
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
        fb = human.getRanking(Ds)
        if obj.n_gen == args.numWarmup and args.numWarmup != 0:
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
      feasibleIndex, scores = getHumanScores(F, w_opt, activeConstraintSet)
      acc = len(feasibleIndex) / len(F)
      print('Feasible ratio: {:.3f}'.format(acc))
      with np.printoptions(formatter={'float': '{: .3f}'.format}):
        print(scores)
      print()
    else:
      print("Exceed maximal number of queries!", end=' ')
      print("Accumulated {:d} feedback".format(n_acc_fb))

    # report(agent, F, w_opt, showRankNumber=10,
    #     activeConstraintSet=activeConstraintSet)

end_time = time.time()
print("It took {:.1f} seconds".format(end_time - start_time))
# endregion

# region: finally obtain the result object
save_obj(agent, os.path.join(outFolder, 'agent'))

res = obj.result()
if args.useTimeStr:
  timestr = time.strftime("%m-%d-%H_%M")
else:
  timestr = ''
picklePath = os.path.join(outFolder, timestr + 'res.pkl')
with open(picklePath, 'wb') as output:
  pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

F = -res.F
fig = plotResultPairwise(
    n_obj, F, objective_names, axis_bound,
    activeConstraintSet=activeConstraintSet
)
fig.tight_layout()
fig.savefig(os.path.join(figFolder, 'objPairwise.png'))

indices = np.argsort(F[:, 0])
F = F[indices]
_F = agent.inference.normalize(F)
objEvalPath = os.path.join(objEvalFolder, 'obj' + str(updateTimes) + '.npy')
np.save(objEvalPath, _F)
feasibleIndex, scores = getHumanScores(_F, w_opt, activeConstraintSet)
acc = len(feasibleIndex) / len(_F)
print()
print('Feasible ratio: {:.3f}'.format(acc))
with np.printoptions(formatter={'float': '{: .3f}'.format}):
  print(_F)
  print(scores)

#== plot output ==
obj_list_un = np.array([-150, -1])
obj_list = (obj_list_un
            - axis_bound[2, 0]) / (axis_bound[2, 1] - axis_bound[2, 0])
levelRatios = np.array([0.25, 0.75])

# plot hyper-parameters
subfigSz = 4
cm = 'coolwarm'
lw = 2.5

# get output
nx = 101
ny = 101
X, Y, Z = getInferenceOutput(agent, nx, ny, obj_list)

# 3D-plot
fig = plotOutput3D(
    X, Y, Z, obj_list_un, axis_bound, fsz=16, subfigSz=subfigSz, cm=cm
)
fig.tight_layout()
figPath = os.path.join(figFolder, 'GP_op_' + args.querySelectorType + '.png')
fig.savefig(figPath)

# 2D-plot
fig2, axes = plotOutput2D(
    X, Y, Z, obj_list_un, axis_bound, levelRatios, fsz=18, subfigSz=subfigSz,
    cm=cm, lw=lw
)
for ax in axes:
  # plot constraint threshold
  ax.plot([0.2, 0.2], [0, 1], 'r:', lw=lw - 0.5)
  ax.plot([0, 1], [0.2, 0.2], 'r:', lw=lw - 0.5)
fig2.tight_layout()
figPath = os.path.join(
    figFolder, 'GP_op_2D_' + args.querySelectorType + '.png'
)
fig2.savefig(figPath)
# endregion
