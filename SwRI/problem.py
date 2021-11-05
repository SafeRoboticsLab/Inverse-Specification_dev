# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Modified from str.py (Perspecta Labs)

import numpy as np
import os
import types


os.sys.path.append('..')
from .flight_dynamics import SWRIFlightDynamics
from pymoo.core.problem import ElementwiseProblem


class SWRIsim(ElementwiseProblem):

  def __init__(self):
    xl = np.zeros(10)
    xl[0] = -5.0

    xu = np.ones(10)
    xu[0] = 5.0

    super().__init__(n_var=10, n_obj=1, n_constr=2, xl=xl, xu=xu)

  def _evaluate(self, x, out, *args, **kwargs):
    out["F"] = np.sum((x - 0.5)**2)
    out["G"] = np.column_stack([0.1 - out["F"], out["F"] - 0.5])
