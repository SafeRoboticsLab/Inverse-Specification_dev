# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC
import numpy as np
from functools import partial
from multiprocessing.dummy import Pool

from .flight_dynamics import SWRIFlightDynamics, SWRIFlightDynamicsParallel
from pymoo.core.problem import ElementwiseProblem, Problem


class SWRIWrapper(ABC):

  def __init__(
      self, values_to_extract=None, objective_names=None, obj_indicator=None
  ):
    """A wrapper to deal with input transformation and output extraction.

    Args:
        values_to_extract (list, optional): which values to extract from the
            FDM outputs. Defaults to None.
        objective_names (dict, optional): the names of the extracted values.
            Defaults to None.
        obj_indicator (np.ndarray, optional): indicators that outputs are
            supposed to be minimize (+1) or maximize (-1). Defaults to None.
    """
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
    self.n_obj = self.values_to_extract.shape[0]

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

  def _output_extracter(
      self, output, get_score=False, get_all=False, **kwargs
  ):
    if get_score or get_all:
      scores = np.array(output['Path_traverse_score_based_on_requirements'])

    if not get_score or get_all:
      y = []
      for key, value in output.items():
        if np.any(self.values_to_extract == key):
          y.append(value)
      features = np.array(y) * self.obj_indicator

    if get_all:
      return features, scores
    elif get_score:
      return scores
    else:
      return features


class SWRISimSerial(SWRIWrapper):

  def __init__(
      self, template_file, exec_file, values_to_extract=None,
      objective_names=None, obj_indicator=None
  ):
    super().__init__(
        values_to_extract=values_to_extract, objective_names=objective_names,
        obj_indicator=obj_indicator
    )
    self.sim = SWRIFlightDynamics(template_file, exec_file)

  def get_fetures(self, x, *args, **kwargs):
    input = self._input_wrapper(x)
    output = self.sim.sim(input, delete_folder=True)
    get_score = False
    if 'get_score' in kwargs:
      get_score = kwargs['get_score']
    return self._output_extracter(output, get_score=get_score)


class SWRISimParallel(SWRIWrapper):

  def __init__(
      self, template_file, exec_file, num_workers, prefix='eval_',
      values_to_extract=None, objective_names=None, obj_indicator=None
  ):
    super().__init__(
        values_to_extract=values_to_extract, objective_names=objective_names,
        obj_indicator=obj_indicator
    )
    self.num_workers = num_workers
    self.sim = SWRIFlightDynamicsParallel(
        template_file, exec_file, self.num_workers, prefix=prefix
    )

  def get_fetures(
      self, X, delete_folder=True, get_score=False, get_all=False, *args,
      **kwargs
  ):
    pool = Pool(self.num_workers)
    input_dict = []
    for input in pool.imap(self._input_wrapper, X):
      input_dict.append(input)
    pool.close()
    pool.join()

    if 'delete_folder' in kwargs:
      delete_folder = kwargs['delete_folder']

    Y = self.sim.sim(input_dict, delete_folder=delete_folder)

    if 'get_score' in kwargs:
      get_score = kwargs['get_score']
    if get_score:
      features = np.empty(shape=(X.shape[0], 1))
    else:
      features = np.empty(shape=(X.shape[0], self.n_obj))

    if 'get_all' in kwargs:
      get_all = kwargs['get_all']
    if get_all:
      scores = np.empty(shape=(X.shape[0], 1))

    pool = Pool(self.num_workers)
    partial_func = partial(
        self._output_extracter, get_score=get_score, get_all=get_all
    )
    if get_all:
      for i, (y, score) in enumerate(pool.imap(partial_func, Y)):
        features[i, :] = y
        scores[i] = score
    else:
      for i, y in enumerate(pool.imap(partial_func, Y)):
        features[i, :] = y
    pool.close()
    pool.join()

    if get_all:
      return features, scores
    else:
      return features


class SWRIElementwiseProblem(ElementwiseProblem, SWRISimSerial):

  def __init__(
      self, template_file, exec_file, values_to_extract=None,
      objective_names=None, obj_indicator=None
  ):

    # SwRI
    SWRISimSerial.__init__(
        self, template_file, exec_file, values_to_extract=values_to_extract,
        objective_names=objective_names, obj_indicator=obj_indicator
    )

    # Pymoo
    xl = np.zeros(len(self.input_names))
    xu = np.array([5., 5., 5., 5., 5., 50.])
    ElementwiseProblem.__init__(
        self, n_var=len(self.input_names), n_obj=len(self.objective_names),
        n_constr=0, xl=xl, xu=xu
    )

  def _evaluate(self, x, out, *args, **kwargs):
    out['F'] = self.get_fetures(x, *args, **kwargs)


class SWRIProblem(Problem, SWRISimParallel):

  def __init__(
      self, template_file, exec_file, num_workers, prefix='eval_',
      values_to_extract=None, objective_names=None, obj_indicator=None
  ):

    # SwRI
    SWRISimParallel.__init__(
        self, template_file, exec_file, num_workers, prefix=prefix,
        values_to_extract=values_to_extract, objective_names=objective_names,
        obj_indicator=obj_indicator
    )

    # Pymoo
    xl = np.zeros(len(self.input_names))
    xu = np.array([5., 5., 5., 5., 5., 50.])
    Problem.__init__(
        self, n_var=len(self.input_names), n_obj=len(self.objective_names),
        n_constr=0, xl=xl, xu=xu
    )

  def _evaluate(self, X, out, *args, **kwargs):
    out['F'], out['scores'] = self.get_fetures(
        X, get_all=True, *args, **kwargs
    )


class SWRIProblemInvSpec(Problem):

  def __init__(
      self, template_file, exec_file, num_workers, inference, prefix='eval_',
      objective_names=dict(o1="InvSpec Score")
  ):

    # SwRI
    self.sim = SWRISimParallel(
        template_file, exec_file, num_workers, prefix=prefix,
        values_to_extract=None, objective_names=None, obj_indicator=None
    )

    # Pymoo
    self.input_names = self.sim.input_names
    self.objective_names = objective_names
    self.input_mask = self.sim.input_mask
    xl = np.zeros(len(self.sim.input_names))
    xu = np.array([5., 5., 5., 5., 5., 50.])
    Problem.__init__(
        self, n_var=len(self.sim.input_names), n_obj=len(self.objective_names),
        n_constr=0, xl=xl, xu=xu
    )

    # InvSpec
    self.inference = inference

  def get_all(self, X):
    features, oracle_scores = self.sim.get_fetures(X, get_all=True)
    # negative sign changes from minimization to maximization
    inputs_to_invspec = self.inference.normalize(-features)
    predicted_scores = self.inference.eval(inputs_to_invspec)
    return -features, oracle_scores.reshape(-1), predicted_scores.reshape(-1)

  def _evaluate(self, X, out, *args, **kwargs):
    features, out['scores'] = self.sim.get_fetures(
        X, get_all=True, *args, **kwargs
    )
    inputs_to_invspec = self.inference.normalize(-features)
    out['F'] = -self.inference.eval(
        inputs_to_invspec
    )  # pymoo wants to minimize
