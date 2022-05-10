# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union, List
import warnings
import numpy as np
from collections import namedtuple

from funct_approx.memory import ReplayMemory
from invspec.design import Design, design2metrics

Feedback = namedtuple('Feedback', ['q_1', 'q_2', 'f'])


class Inference(ABC):

  def __init__(
      self, state_dim: int, action_dim: int, CONFIG: Any, key: str,
      input_min: Optional[np.ndarray] = None,
      input_max: Optional[np.ndarray] = None, input_normalize: bool = True
  ) -> None:

    super().__init__()
    self.update_times = 0
    #= ENV
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.input_min = input_min
    self.input_max = input_max
    self.input_normalize = input_normalize
    self.key = key  # how to retrive features from designs

    #= OBJECTS
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)

  #region: == Interface with a Design Exploration Engine ==
  def design2input(
      self, designs: Union[List[Design], np.ndarray]
  ) -> np.ndarray:
    """
    Transforms designs into metrics based on forward features (from the
        simulator). The self.key specifies which test run to retrive.
        Normalizes the metrics if specified when constructed.

    Args:
        designs (Union[List[Design], np.ndarray]): a list of Designs or
            retrieved metrics.

    Returns:
        np.ndarray: appropriate inputs for the downstream inference engine.
    """
    if isinstance(designs, np.ndarray):
      inputs = designs
    else:
      inputs = design2metrics(designs, self.key)
    if self.input_normalize:  # normalize by a priori input_min and input_max
      inputs = self.normalize(inputs)

    return inputs.astype('float32')

  def eval(self, pop: Union[List[Design], np.ndarray],
           **kwargs) -> Union[List[Design], np.ndarray]:
    """
    A wrapper for fitness evaluation. If the designs are presented in the
    format of Design, we extract the forward metrics and normalize it
    (optional).

    Args:
        pop: The population which should be evaluated.

    Returns:
        np.ndarray: float, fitness of the designs in the current population
    """
    input = self.design2input(pop)
    fitness = self._eval(input, **kwargs)

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
      self, query: Union[List[Design], np.ndarray], **kwargs
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
