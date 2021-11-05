# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
import numpy as np

from collections import namedtuple


Feedback = namedtuple('Feedback', ['q_1', 'q_2', 'f'])

from funct_approx.memory import ReplayMemory

from pymoo.core.population import Population


class Inference(ABC):

  def __init__(
      self, stateDim, actionDim, CONFIG, F_min=None, F_max=None,
      F_normalize=True
  ):

    super().__init__()
    #= ENV
    self.stateDim = stateDim
    self.actionDim = actionDim
    self.F_min = F_min
    self.F_max = F_max
    self.F_normalize = F_normalize

    #= OBJECTS
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)

  #region: == Interface with GA ==
  def design2obj(self, designs):
    if isinstance(designs, Population):
      # currently only interact with objectives:
      # add '-' because we want to maximize, but GA wants to minimize
      if self.F_normalize:  # normalize by a priori F_min anf F_max
        F = self.normalize(-designs.get('F'))
      else:
        F = -designs.get('F')
    elif isinstance(designs, np.ndarray):
      F = designs
    else:
      raise ValueError(
          "Designs must be either pymoo:Population or numpy:array!"
      )
    return F

  def eval(self, pop, **kwargs):
    """
    A wrapper for fitness evaluation. If the designs are presented in the
    format of Pymoo:population, we extract the obejectives and normalize if
    needed.

    Args:
        pop: The population which should be evaluated.

    Returns:
        np.ndarray: float, fitness of the designs in the current population
    """
    F = self.design2obj(pop)
    fitness = self._eval(F, **kwargs)

    if isinstance(pop, Population):
      for i, ind in enumerate(pop):
        ind.set("fitness", fitness[i])
      return pop
    elif isinstance(pop, np.ndarray):
      return fitness

  @abstractmethod
  def _eval(self, F, **kwargs):
    """
    Evaluates the fitness according to the (normalized) obejective
    measurements. The child class must implement this function.

    Args:
        np.ndarray: matrix of obejective measurements, of shape
            (#designs, #obj)
    """
    raise NotImplementedError

  def eval_query(self, query, **kwargs):
    """
    Evaluates the query. For example, the evaluation can base on information
    gain or value of information

    Args:
        query: a pair of designs, of shape (2,)
    """
    F = self.design2obj(query)
    metric = self._eval_query(F, **kwargs)
    return metric

  @abstractmethod
  def _eval_query(self, F, **kwargs):
    """
    Evaluate the quality of the query. The child class must implement this
    function.

    Args:
        np.ndarray: matrix of obejective measurements, of shape (2, #obj)
    """
    raise NotImplementedError

  #endregion

  #region: == LEARN ==
  @abstractmethod
  def learn(self):
    raise NotImplementedError

  @abstractmethod
  def initialize(self):
    raise NotImplementedError

  #endregion

  #region: == MEMORY ==
  def store_feedback(self, *args):
    self.memory.update(Feedback(*args))

  def clear_feedback(self):
    self.memory.reset()

  #endregion

  #region: == Utils ==
  def normalize(self, F):
    F_spacing = self.F_max - self.F_min

    return (F - self.F_min) / F_spacing

  def get_all_query_feedback(self):
    feedbacks = self.memory.memory
    batch = Feedback(*zip(*feedbacks))

    return self.extract_batch(batch)

  def get_sampled_query_feedback(self, size):
    feedbacks = self.memory.sample(size)
    batch = Feedback(*zip(*feedbacks))

    return self.extract_batch(batch)

  @classmethod
  def extract_batch(cls, batch):
    """
    Extracts batch of Feedback into two trajectories arrays and a feedback
    array.

    Args:
        batch (object): consists of q_1, q_2 and f.

    Returns:
        tuple of ndarrays: trajectories_1, trajectories_2, feedback
    """
    q_1s = batch.q_1
    q_2s = batch.q_2
    trajectories_1 = cls.trans_state_action_tuple2numpy(q_1s)
    trajectories_2 = cls.trans_state_action_tuple2numpy(q_2s)

    return trajectories_1, trajectories_2, np.array(batch.f)

  @staticmethod
  def trans_state_action_tuple2numpy(qs):
    """
    Transforms a sequence of (state, action) pairs into a numpy array of shape
    (batchsize, length, stateDim+actionDim).

    Args:
        qs (tuple): tuple of (state, action) tuples.

    Returns:
        np.ndarray: of shape (batchsize, length, stateDim+actionDim).
    """
    batchsize = len(qs)
    length = qs[0][0].shape[0]
    stateDim = qs[0][0].shape[1]
    actionDim = qs[0][1].shape[1]

    trajectories = np.empty(shape=(batchsize, length, stateDim + actionDim))

    for i, q in enumerate(qs):
      state, action = q
      trajectories[i, :, :stateDim] = state
      trajectories[i, :, stateDim:] = action

    return trajectories

  #endregion
