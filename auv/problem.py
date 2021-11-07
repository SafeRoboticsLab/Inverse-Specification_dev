# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Modified from str.py (Perspecta Labs)

import numpy as np
import os
import types

os.sys.path.append('..')
from .auv_sim import DexcelInterface_p1, DexcelInterface_p2, DexcelInterface_p3
from pymoo.core.problem import ElementwiseProblem


def build_forward_funct(problem_type='p1'):
  # types.SimpleNamespace provides a mechanism to instantiate an object that
  # can hold attributes and nothing else. It is, in effect, an empty class
  # with a fancier __init__() and a helpful __repr__().
  fparams = types.SimpleNamespace()
  if problem_type == 'p1':
    fparams.func = DexcelInterface_p1()
  elif problem_type == 'p2':
    fparams.func = DexcelInterface_p2()
  elif problem_type == 'p3':
    fparams.func = DexcelInterface_p3()
  else:
    raise ValueError(
        "Unsupported problem type (problem_type),"
        + "we support p1 and p2 now!"
    )
  fparams.xcount = len(fparams.func.inputs)

  fparams.bounds = {
      # 'Diameter': (.25, 1.00),
      'Diameter': (0, 8),
      'Length': (1.0, 5.0),
      # 'PV_Material_Choice': (1, 5),
      'PV_Material_Choice': (0, 4),
      'Depth_Rating': (200., 500.),
      'Safety_Factor_for_Depth_Rating': (1.33, 2.0),
      # 'Battery_Specific_Energy': (100., 360.),
      'Battery_Specific_Energy': (0, 4),
      'Battery_Fraction': (0.4, 0.6),
      # 'Design_Hotel_Power_Draw': (20., 22.),
      'Design_Hotel_Power_Draw': (0, 14),
      'Cd_Drag_coefficient': (0.0078, 0.0080),
      'Appendage_added_area': (0.1, 0.2),
      'Propulsion_efficiency': (0.45, 0.55),
      'Density_of_seawater': (1025., 1030.),
      'CruiseSpeed': (0.5, 1.5)
  }
  assert len(fparams.bounds) == fparams.xcount, \
      "#bounds ({}) doesn't match #inputs ({})".format(
          len(fparams.bounds), fparams.xcount)

  fparams.discrete = {
      'Diameter': [0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.75, 0.8, 1.0],
      'PV_Material_Choice': [1, 2, 3, 4, 5],
      'Battery_Specific_Energy': [100., 125., 200., 250., 360.],
      'Design_Hotel_Power_Draw': [
          20., 20.25, 20.5, 20.75, 21., 21.25, 21.5, 21.75, 22., 40., 41., 42.,
          43., 44., 45.
      ]
  }

  fparams.mask = [
      "int", "real", "int", "real", "real", "int", "real", "int", "real",
      "real", "real", "real", "real"
  ]
  assert len(fparams.mask) == fparams.xcount, \
      "The length of mask ({}) doesn't match #inputs ({})".format(
          len(fparams.mask), fparams.xcount)

  # Collect the inputs
  fparams.xinputs = dict()
  i = 0
  for var_name in fparams.func.inputs:
    if var_name not in fparams.bounds:
      if var_name not in fparams.one_hots:
        print(f"Woopsie, {var_name} doesn't have a bounds")
      continue
    fparams.xinputs[var_name] = i
    i += 1

  # Get LB and UB as vectors
  fparams.xl = np.empty(shape=(len(fparams.bounds),), dtype=np.double)
  fparams.xu = np.empty(shape=(len(fparams.bounds),), dtype=np.double)
  for i, value in enumerate(fparams.bounds.values()):
    fparams.xl[i] = value[0]
    fparams.xu[i] = value[1]

  # Get our counts and indices for later
  fparams.numobjs = len(fparams.func.objectives)
  fparams.numconsts = len(fparams.func.constraints)
  fparams.ycount = fparams.numobjs + fparams.numconsts
  # index of the first constraint in the output of ff()
  fparams.indconsts = len(fparams.func.objectives)
  return fparams


class AUVsim(ElementwiseProblem):

  def __init__(self, problem_type='p1'):
    self.problem_type = problem_type
    print("Problem type: {}".format(self.problem_type))
    self.fparams = build_forward_funct(problem_type=self.problem_type)
    super().__init__(
        n_var=self.fparams.xcount, n_obj=self.fparams.numobjs,
        n_constr=self.fparams.numconsts, xl=self.fparams.xl, xu=self.fparams.xu
    )

  def _generateDictFromIndividual(self, ind):
    xDict = {}
    for i, var_name in enumerate(self.fparams.func.inputs.keys()):
      if var_name in (self.fparams.discrete.keys()):
        xDict[var_name] = self.fparams.discrete[var_name][int(ind[i])]
      else:
        xDict[var_name] = ind[i]
    return xDict

  # spreadsheet only supports elementwise evaluation
  def _evaluate(self, x, out, *args, **kwargs):
    evaluation = np.array(
        self.fparams.func.problem(self._generateDictFromIndividual(x))
    )

    # The default is to minimize
    out["F"] = [-f for f in evaluation[:self.fparams.indconsts]]
    out["G"] = [g for g in evaluation[self.fparams.indconsts:]]
