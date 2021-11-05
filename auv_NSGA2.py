# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
# example: python3 auv_NSGA2.py -p p1 -ng 200 -psz 100
# example: python3 auv_NSGA2.py -cg 1

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time


timestr = time.strftime("%m-%d-%H_%M")
import os


os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from auv.problem import AUVsim
from utils import setSeed, plotResult3D, plotResultPairwise

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
parser.add_argument(
    "-ng", "--numGen", help="#generation", default=200, type=int
)
parser.add_argument(
    "-cg", "--checkGeneration", help="check period", default=25, type=int
)
parser.add_argument(
    "-o", "--optimizer", help="problem type", default='oo', type=str,
    choices=['oo', 'built-in']
)

args = parser.parse_args()
print(args)
outFolder = os.path.join('scratch', 'AUV_' + args.problemType, 'NSGA2')
figFolder = os.path.join(outFolder, 'figure')
os.makedirs(figFolder, exist_ok=True)
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
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
)
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

algorithm = NSGA2(
    pop_size=args.popSize,
    n_offsprings=args.popSize,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
)
# termination criterion
termination = get_termination("n_gen", args.numGen)
# endregion

# region: == Define Solver ==
print("\n== Optimization starts ...")
if args.optimizer == 'oo':
  # perform a copy of the algorithm to ensure reproducibility
  import copy
  obj = copy.deepcopy(algorithm)

  # let the algorithm know what problem we are intending to solve and provide
  # other attributes
  obj.setup(
      problem, termination=termination, seed=args.randomSeed, save_history=True
  )

  # until the termination criterion has not been met
  while obj.has_next():
    print(obj.n_gen, end='\r')
    # perform an iteration of the algorithm
    obj.next()

    # check performance
    if (obj.n_gen - 1) % args.checkGeneration == 0:
      n_gen = obj.n_gen
      n_nds = len(obj.opt)
      CV = obj.opt.get('CV').min()
      print(f"gen[{n_gen}]: n_nds: {n_nds} CV: {CV}")
      F = -obj.opt.get('F')
      if n_obj == 3:
        fig = plotResult3D(F, objective_names, axis_bound)
      else:
        fig = plotResultPairwise(n_obj, F, objective_names, axis_bound)
      fig.supxlabel(str(n_gen), fontsize=20)
      fig.tight_layout()
      figProFolder = os.path.join(figFolder, 'progress')
      os.makedirs(figProFolder, exist_ok=True)
      fig.savefig(os.path.join(figProFolder, str(n_gen) + '.png'))
      plt.close()
      # idx = np.argmax(F[:, 0])
      # for i, tmp in enumerate(F[idx]):
      #     if i == n_obj-1:
      #         print(tmp)
      #     else:
      #         print(tmp, end=', ')

  # finally obtain the result object
  res = obj.result()
elif args.optimizer == 'built-in':
  # get results via built-in minimization
  res = minimize(
      problem, algorithm, termination, seed=args.randomSeed, save_history=True,
      verbose=False
  )

  #== Result ==
  X_his = []
  F = []
  CV = []

  for alg_tmp in res.history:
    X_his.append(alg_tmp.pop.get("X"))
    F.append(alg_tmp.pop.get("F"))
    CV.append(alg_tmp.pop.get("CV"))

  fsz = 16
  nRow = 5
  nCol = 5
  space = np.floor(args.numGen / (nRow*nCol))
  subfigSz = 3
  figsize = (nCol * subfigSz, nRow * subfigSz)

  objIdx1 = 0
  objIdx2 = 1

  fig, axArray = plt.subplots(nRow, nCol, figsize=figsize)

  for i in range(nRow):
    for j in range(nCol):
      idx = int((i*nCol + j + 1) * space - 1)
      print(idx, end=', ')
      ax = axArray[i][j]
      ax.scatter(-F[idx][:, objIdx1], -F[idx][:, objIdx2], c='b', s=6)

      ax.set_title(str(idx), fontsize=fsz)
      ax.set_xlim(0, 12000)
  #         ax.set_ylim(0.4, 1.6)

  fig.supxlabel(
      problem.fparams.func.objective_names['o' + str(objIdx1 + 1)],
      fontsize=fsz
  )
  fig.supylabel(
      problem.fparams.func.objective_names['o' + str(objIdx2 + 1)],
      fontsize=fsz
  )

  fig.tight_layout()
  figPath = os.path.join(
      figFolder, args.problemType + '-optimal_front_all.png'
  )
  fig.savefig(figPath)

  numSnapshot = int(problem.n_obj * (problem.n_obj - 1) / 2)
  if numSnapshot < 5:
    nCol = numSnapshot
  else:
    nCol = 5
  nRow = int(np.ceil(numSnapshot / nCol))
  subfigSz = 4
  figsize = (nCol * subfigSz, nRow * subfigSz)

  fig, axArray = plt.subplots(nRow, nCol, figsize=figsize)
  fsz = 16
  sz = 20

  idx = 0
  for i in range(problem.n_obj):
    for j in range(i + 1, problem.n_obj):
      rowIdx = int(idx / nCol)
      colIdx = idx % nCol
      if nRow > 1:
        ax = axArray[rowIdx, colIdx]
      elif nCol > 1:
        ax = axArray[colIdx]
      else:
        ax = axArray
      ax.scatter(-res.F[:, i], -res.F[:, j], c='b', s=sz)
      ax.set_xlabel(
          problem.fparams.func.objective_names['o' + str(i + 1)], fontsize=fsz
      )
      ax.set_ylabel(
          problem.fparams.func.objective_names['o' + str(j + 1)], fontsize=fsz
      )
      idx += 1
  fig.tight_layout()
  figPath = os.path.join(figFolder, args.problemType + '-optimal_front.png')
  fig.savefig(figPath)
# endregion

# region: Check with DexcelInterface
print("\nCheck with DexcelInterface")
from auv.auv_sim import (
    DexcelInterface_p1, DexcelInterface_p2, DexcelInterface_p3
)


np.set_printoptions(precision=2)
if problem.problemType == 'p1':
  ff_obj = DexcelInterface_p1()
elif problem.problemType == 'p2':
  ff_obj = DexcelInterface_p2()
elif problem.problemType == 'p3':
  ff_obj = DexcelInterface_p3()

for i in range(1):
  x = res.X[i]
  print('Individual', i)
  f = res.F[i]
  g = res.G[i]
  print("individual used in GA", x)
  xDict = problem._generateDictFromIndividual(x)
  print("individual used to access the interface", xDict)

  out = ff_obj.problem(xDict)
  print("Obj. and Constr. by the interface")
  for tmp in out:
    print('{:.2f}'.format(tmp), end=', ')
  print()
  print("Obj. and Constr. obtained in the wrapper")
  for tmp in f:
    print('{:.2f}'.format(tmp), end=', ')
  for tmp in g:
    print('{:.2f}'.format(tmp), end=', ')
  print()
  print('-' * 80)
# endregion

# region: finally obtain the result object
res = obj.result()
picklePath = os.path.join(outFolder, timestr + '.pkl')
with open(picklePath, 'wb') as output:
  pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
print(picklePath)

F = -res.F
fig = plotResultPairwise(
    n_obj, F, objective_names, axis_bound, nColDefault=5, subfigSz=4, fsz=16,
    sz=20
)
fig.tight_layout()
fig.savefig(os.path.join(figFolder, 'objPairwise.png'))

print("\npick the design in the optimal front that has maximal objective 1.")
indices = np.argsort(F[:, 0])
F = F[indices]
with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
  print(F[-1])


def normalize(F, F_max, F_min):
  F_spacing = F_max - F_min
  return (F-F_min) / F_spacing


_F = normalize(F, F_max, F_min)
with np.printoptions(formatter={'float': '{: 2.2f}'.format}):
  print(_F)
# endregion
