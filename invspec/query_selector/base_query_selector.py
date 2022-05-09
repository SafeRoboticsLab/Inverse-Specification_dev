# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from pymoo.core.population import Population


class QuerySelector(ABC):

  def __init__(self) -> None:
    """
    This class is used to select queries to send to the human designer.
    Several strategies can be used.
    """
    super().__init__()
    self.num_query_times = 0  # number of query times (different to #queries)

  def do(
      self, pop: Union[Population, np.ndarray], n_queries: int, n_designs: int,
      **kwargs
  ) -> np.ndarray:
    """Choose from the population new individuals to be selected.

    Args:
        pop (:class:`~pymoo.core.population.Population`): The population
            which should be selected from. Some criteria from the design or
            objective space might be used for the selection. Therefore,
            only the number of individual might be not enough.
        n_queries (int): Number of queries to send.
        n_designs (int): Number of designs in each query.

    Returns:
        ndarray: Indices of selected individuals.
    """
    max_n_queries = int(len(pop) * (len(pop) - 1) / 2)
    if n_queries > max_n_queries:
      n_queries_obtained = max_n_queries
      print("Warning: can only get {} queries".format(max_n_queries))
    else:
      n_queries_obtained = n_queries
    I = self._do(pop, n_queries_obtained, n_designs, **kwargs)
    self.num_query_times += 1

    return I

  @abstractmethod
  def _do(
      self, pop: Union[Population, np.ndarray], n_queries: int, n_designs: int,
      **kwargs
  ) -> np.ndarray:
    raise NotImplementedError
