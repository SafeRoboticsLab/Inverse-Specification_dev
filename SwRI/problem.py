# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
import numpy as np
from functools import partial
from multiprocessing.dummy import Pool

from .flight_dynamics import SWRIFlightDynamics, SWRIFlightDynamicsParallel
from pymoo.core.problem import ElementwiseProblem, Problem


class SWRIProblem(ABC):

  def __init__(
      self, values_to_extract=None, objective_names=None, obj_indicator=None
  ):
    if objective_names is None:
      self.values_to_extract = np.array([
          "Flight_distance",
          "Time_to_traverse_path",
          "Average_speed_to_traverse_path",
          "Maximimum_error_distance_during_flight",
          # "Time_of_maximum_distance_error",
          "Spatial_average_distance_error",
          # "Maximum_ground_impact_speed",
      ])
      self.objective_names = dict(
          o1="Distance",
          o2="Time",
          o3="Speed_avg",
          o4="Dist_err_max",
          # o5="Time_inst_max_err",
          o5="Dist_err_avg",
          # o7="Gnd_speed_max",
      )
      self.obj_indicator = np.array([-1., 1., -1., 1., 1.])  # -1: max; +1: min
    else:
      self.values_to_extract = values_to_extract
      self.objective_names = objective_names
      self.obj_indicator = obj_indicator
    self.input_names = np.array([
        'control_Q_position',
        'control_Q_velocity',
        'control_Q_angular_velocity',
        'control_Q_angles',
        'control_R',
        'control_requested_lateral_speed',
    ])
    self.input_mask = ['real', 'real', 'real', 'real', 'real', 'real']

  def _input_wrapper(self, x):
    """Wraps designs to fit in the format of the simulator input.

    Args:
        x (np.ndarray): features of a design

    Returns:
        dict: an input for the simulator.
    """
    input = dict(
        # region: fixed parameters
        control_iaileron=5,
        control_iflap=6,
        control_i_flight_path=5,
        control_requested_vertical_speed=0,
        # endregion
        # region: inputs, LQR parameters
        control_Q_position=x[0],
        control_Q_velocity=x[1],
        control_Q_angular_velocity=x[2],
        control_Q_angles=x[3],
        control_R=x[4],
        control_requested_lateral_speed=x[5],
        # endregion
    )
    return input

  def _output_extracter(self, output, get_score=False, **kwargs):
    if get_score:
      return np.array(output['Path_traverse_score_based_on_requirements'])
    else:
      y = []
      for key, value in output.items():
        if np.any(self.values_to_extract == key):
          y.append(value)

    return np.array(y) * self.obj_indicator

  @abstractmethod
  def _evaluate(self, x, out, *args, **kwargs):
    raise NotImplementedError


class SWRISimulator(ElementwiseProblem, SWRIProblem):

  def __init__(
      self, template_file, exec_file, values_to_extract=None,
      objective_names=None, obj_indicator=None
  ):

    # SwRI
    SWRIProblem.__init__(
        self, values_to_extract=values_to_extract,
        objective_names=objective_names, obj_indicator=obj_indicator
    )
    self.sim = SWRIFlightDynamics(template_file, exec_file)

    # Pymoo
    xl = np.zeros(len(self.input_names))
    xu = np.array([5., 5., 5., 5., 5., 50.])
    Problem.__init__(
        self, n_var=len(self.input_names), n_obj=len(self.objective_names),
        n_constr=0, xl=xl, xu=xu
    )

  def _evaluate(self, x, out, *args, **kwargs):
    input = self._input_wrapper(x)
    output = self.sim.sim(input, delete_folder=True)
    get_score = False
    if 'get_score' in kwargs:
      get_score = kwargs['get_score']
    out["F"] = self._output_extracter(output, get_score=get_score)


class SWRISimulatorParallel(Problem, SWRIProblem):

  def __init__(
      self, template_file, exec_file, num_workers, values_to_extract=None,
      objective_names=None, obj_indicator=None
  ):

    # SwRI
    SWRIProblem.__init__(
        self, values_to_extract=values_to_extract,
        objective_names=objective_names, obj_indicator=obj_indicator
    )
    # parallel
    self.num_workers = num_workers
    # self.sim = SWRIFlightDynamics(template_file, exec_file)
    self.sim = SWRIFlightDynamicsParallel(
        template_file, exec_file, self.num_workers
    )

    # Pymoo
    xl = np.zeros(len(self.input_names))
    xu = np.array([5., 5., 5., 5., 5., 50.])
    Problem.__init__(
        self, n_var=len(self.input_names), n_obj=len(self.objective_names),
        n_constr=0, xl=xl, xu=xu
    )

  # def _evaluate_individual(self, x, **kwargs):
  #   input = self._input_wrapper(x)
  #   output = self.sim.sim(input, delete_folder=True)
  #   get_score = False
  #   if 'get_score' in kwargs:
  #     get_score = kwargs['get_score']
  #   return self._output_extracter(output, get_score=get_score)

  # def _evaluate(self, X, out, *args, **kwargs):
  #   partial_func = partial(self._evaluate_individual, **kwargs)
  #   get_score = False
  #   if 'get_score' in kwargs:
  #     get_score = kwargs['get_score']
  #   if get_score:
  #     out["F"] = np.empty(shape=(X.shape[0], 1))
  #   else:
  #     out["F"] = np.empty(shape=(X.shape[0], self.n_obj))
  #   pool = Pool(self.num_workers)
  #   for i, y in enumerate(pool.imap(partial_func, X)):
  #     out["F"][i, :] = y

  def _evaluate(self, X, out, *args, **kwargs):
    pool = Pool(self.num_workers)
    input_dict = []
    for i, input in enumerate(pool.imap(self._input_wrapper, X)):
      input_dict.append(input)
    pool.close()
    pool.join()

    delete_folder = True
    if 'delete_folder' in kwargs:
      delete_folder = kwargs['delete_folder']

    Y = self.sim.sim(input_dict, delete_folder=delete_folder)

    get_score = False
    if 'get_score' in kwargs:
      get_score = kwargs['get_score']
    if get_score:
      out["F"] = np.empty(shape=(X.shape[0], 1))
    else:
      out["F"] = np.empty(shape=(X.shape[0], self.n_obj))

    pool = Pool(self.num_workers)
    partial_func = partial(self._output_extracter, get_score=get_score)
    for i, y in enumerate(pool.imap(partial_func, Y)):
      out["F"][i, :] = y
    pool.close()
    pool.join()
