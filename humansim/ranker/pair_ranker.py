# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import abstractmethod
from typing import List, Tuple, Optional, Any
import numpy as np

from humansim.ranker.ranker import Ranker
from utils import get_infeasible_designs
from invspec.design import Design, design2metrics, design2params


class PairRanker(Ranker):

  def __init__(
      self, beta: float = 5.,
      active_constraint_set: Optional[List[Tuple[str, float]]] = None,
      perfect_rank: bool = False, indifference: float = 0.1
  ):

    super().__init__()
    self.beta = beta
    self.consider_constarint = (active_constraint_set is not None)
    self.active_constraint_set = active_constraint_set
    self.perfect_rank = perfect_rank
    self.indifference = indifference

  def _get_ranking(self, query: List[Design], **kwargs) -> int:
    """Gets the preference of designs or returns "cannot distinguish".

    Args:
        query (List[Design]): the length should be 2.

    Returns:
        int: feedback, 0: first, 1: second, 2: cannot distinguish.
    """
    n_designs = len(query)
    assert n_designs == 2, "This ranker only supports binary preference!"

    if self.consider_constarint:
      indicator = get_infeasible_designs(
          design2metrics(query, key=kwargs['key']), self.active_constraint_set
      )
      feasible_index = np.arange(n_designs)[np.logical_not(indicator)]
    else:
      feasible_index = np.arange(n_designs)

    if feasible_index.shape[0] == 0:
      feedback = 2
    elif feasible_index.shape[0] == 1:
      feedback = feasible_index[0]
    else:
      scores = self._get_scores(query, feasible_index, **kwargs)
      order = feasible_index[np.argsort(-scores)]
      score_diff = np.clip(self.beta * (scores[0] - scores[1]), -20, 20)
      prob = 1 / (1 + np.exp(score_diff))
      if self.perfect_rank:
        feedback = order[0]
      else:
        if np.abs(prob - 0.5) < self.indifference:
          feedback = 2
        else:
          feedback = order[0]

    return feedback

  @abstractmethod
  def _get_scores(
      self, query: List[Design], feasible_index: np.ndarray, **kwargs
  ):
    raise NotImplementedError


class PairRankerWeights(PairRanker):

  def __init__(
      self, w_opt: np.ndarray, beta: float = 5.,
      active_constraint_set: Optional[List[Tuple[str, float]]] = None,
      perfect_rank: bool = False, indifference: float = 0.1
  ):
    super().__init__(
        beta=beta, active_constraint_set=active_constraint_set,
        perfect_rank=perfect_rank, indifference=indifference
    )
    self.w_opt = w_opt

  def _get_scores(
      self, query: List[Design], feasible_index: np.ndarray, **kwargs
  ):
    designs = design2metrics(query, key=kwargs['key'])[feasible_index, :]
    assert designs.shape[1] == self.w_opt.shape[0], (
        "#features ({}) doesn't match #weights ({}).".format(
            designs.shape[1], self.w_opt.shape[0]
        )
    )
    return designs @ self.w_opt


class PairRankerSimulator(PairRanker):

  def __init__(
      self, simulator: Any, beta: float = 5.,
      active_constraint_set: Optional[List[Tuple[str, float]]] = None,
      perfect_rank: bool = False, indifference: float = 0.1
  ):
    super().__init__(
        beta=beta, active_constraint_set=active_constraint_set,
        perfect_rank=perfect_rank, indifference=indifference
    )
    self.sim = simulator

  def _get_scores(
      self, query: List[Design], feasible_index: np.ndarray, **kwargs
  ):
    designs = design2params(query, key=kwargs['key'])[feasible_index, :]
    scores = self.sim.get_fetures(designs, get_score=True, **kwargs)
    scores = scores.reshape(-1)
    print('scores:', scores)
    return scores
