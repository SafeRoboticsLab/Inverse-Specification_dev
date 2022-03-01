# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import warnings
import numpy as np
from collections import namedtuple
from pymoo.core.population import Population

from funct_approx.memory import ReplayMemory

Feedback = namedtuple('Feedback', ['q_1', 'q_2', 'f'])


class Inference(ABC):

  def __init__(
      self, state_dim: int, action_dim: int, CONFIG: Any,
      input_min: Optional[np.ndarray] = None,
      input_max: Optional[np.ndarray] = None, input_normalize: bool = True,
      pop_extract_type: str = 'F'
  ) -> None:

    super().__init__()
    #= ENV
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.input_min = input_min
    self.input_max = input_max
    self.input_normalize = input_normalize
    self.pop_extract_type = pop_extract_type

    #= OBJECTS
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)

  #region: == Interface with GA ==
  def design2input(self, designs: Union[Population, np.ndarray]) -> np.ndarray:
    if isinstance(designs, Population):
      if self.pop_extract_type == 'F':
        # add '-' because we want to maximize, but GA wants to minimize
        ind = -1.
      elif self.pop_extract_type == 'X':
        ind = 1.
      else:
        raise ValueError(
            "The pop_extract_type ({}) is not supported".format(
                self.pop_extract_type
            )
        )
      input = ind * designs.get(self.pop_extract_type)
    elif isinstance(designs, np.ndarray):
      input = designs
    else:
      raise ValueError(
          "Designs must be either pymoo:Population or numpy:array!"
      )
    if self.input_normalize:  # normalize by a priori input_min and input_max
      input = self.normalize(input)

    return input.astype('float32')

  def eval(self, pop: Union[Population, np.ndarray],
           **kwargs) -> Union[Population, np.ndarray]:
    """
    A wrapper for fitness evaluation. If the designs are presented in the
    format of Pymoo:population, we extract the obejectives and normalize if
    needed.

    Args:
        pop: The population which should be evaluated.

    Returns:
        np.ndarray: float, fitness of the designs in the current population
    """
    input = self.design2input(pop)
    fitness = self._eval(input, **kwargs)

    if isinstance(pop, Population):
      for i, ind in enumerate(pop):
        ind.set("fitness", fitness[i])
      return pop
    elif isinstance(pop, np.ndarray):
      return fitness

  @abstractmethod
  def _eval(self, input: np.ndarray, **kwargs) -> np.ndarray:
    """
    Evaluates the fitness according to the (normalized) obejective
    measurements. The child class must implement this function.

    Args:
        np.ndarray: matrix of obejective measurements, of shape
            (#designs, #obj)
    """
    raise NotImplementedError

  def eval_query(
      self, query: Union[Population, np.ndarray], **kwargs
  ) -> float:
    """
    Evaluates the query. For example, the evaluation can base on information
    gain or value of information

    Args:
        query: a pair of designs.
    """
    input = self.design2input(query)
    metric = self._eval_query(input, **kwargs)
    return metric

  @abstractmethod
  def _eval_query(self, input: np.ndarray, **kwargs) -> float:
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
  def learn(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def initialize(self) -> None:
    raise NotImplementedError

  #endregion

  #region: == MEMORY ==
  def store_feedback(self, *args) -> None:
    self.memory.update(Feedback(*args))

  def clear_feedback(self) -> None:
    self.memory.reset()

  #endregion

  #region: == Utils ==
  def normalize(self, input: np.ndarray) -> np.ndarray:
    if (self.input_min is None or self.input_max is None):
      warnings.warn("Need to provide input bounds if using normalize")
      return input
    else:
      input_spacing = self.input_max - self.input_min
      return (input - self.input_min) / input_spacing

  def get_all_query_feedback(
      self
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feedbacks = self.memory.memory
    batch = Feedback(*zip(*feedbacks))

    return self.extract_batch(batch)

  def get_sampled_query_feedback(
      self, size: int
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feedbacks = self.memory.sample(size)
    batch = Feedback(*zip(*feedbacks))

    return self.extract_batch(batch)

  @classmethod
  def extract_batch(cls,
                    batch: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
  def trans_state_action_tuple2numpy(qs: Tuple[Tuple, ...]) -> np.ndarray:
    """
    Transforms a sequence of (state, action) pairs into a numpy array of shape
    (batch_size, length, state_dim+action_dim).

    Args:
        qs (Tuple): tuple of (state, action) tuples.

    Returns:
        np.ndarray: of shape (batch_size, length, state_dim+action_dim).
    """
    batch_size = len(qs)
    length = qs[0][0].shape[0]
    state_dim = qs[0][0].shape[1]
    action_dim = qs[0][1].shape[1]

    trajectories = np.empty(shape=(batch_size, length, state_dim + action_dim))

    for i, q in enumerate(qs):
      state, action = q
      trajectories[i, :, :state_dim] = state
      trajectories[i, :, state_dim:] = action

    return trajectories

  #endregion
