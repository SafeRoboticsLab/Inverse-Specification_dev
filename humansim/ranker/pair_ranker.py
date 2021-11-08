# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
import numpy as np

from humansim.ranker.ranker import Ranker
from utils import get_infeasible_designs


class PairRanker(Ranker):

  def __init__(
      self, beta=5., active_constraint_set=None, perfect_rank=False,
      indifference=0.1
  ):

    super().__init__()
    self.beta = beta
    self.consider_constarint = (active_constraint_set is not None)
    self.active_constraint_set = active_constraint_set
    self.perfect_rank = perfect_rank
    self.indifference = indifference

  def _get_ranking(self, query, **kwargs):
    n_designs, _ = query.shape
    assert n_designs == 2, "This ranker only supports binary preference!"

    if self.consider_constarint:
      indicator = get_infeasible_designs(query, self.active_constraint_set)
      # infeasible_index = np.arange(query.shape[0])[indicator]
      feasible_index = np.arange(n_designs)[np.logical_not(indicator)]
      feasible_designs = query[feasible_index]
      scores = self._get_scores(feasible_designs)
      order_first_part = feasible_index[np.argsort(-scores)]

      if feasible_designs.shape[0] == 0:
        feedback = 2
      elif feasible_designs.shape[0] == 1:
        feedback = feasible_index[0]
      else:
        p = 1 / (1 + np.exp(self.beta * (scores[0] - scores[1])))
        if self.perfect_rank:
          feedback = order_first_part[0]
        else:
          if np.abs(p - 0.5) < self.indifference:
            feedback = 2
          else:
            feedback = order_first_part[0]
    else:
      scores = self._get_scores(query)
      order = np.argsort(-scores)

      p = 1 / (1 + np.exp(self.beta * (scores[0] - scores[1])))
      if self.perfect_rank:
        feedback = order[0]
      else:
        if np.abs(p - 0.5) < self.indifference:
          feedback = 2
        else:
          feedback = order[0]

    return feedback

  @abstractmethod
  def _get_scores(self, designs):
    raise NotImplementedError


class PairRankerWeights(PairRanker):

  def __init__(
      self, w_opt, beta=5., active_constraint_set=None, perfect_rank=False,
      indifference=0.1
  ):
    super().__init__(
        beta=beta, active_constraint_set=active_constraint_set,
        perfect_rank=perfect_rank, indifference=indifference
    )
    self.w_opt = w_opt

  def _get_scores(self, designs):
    assert designs.shape[1] == self.w_opt.shape[0], (
        "#features ({}) doesn't match #weights ({}).".format(
            designs.shape[1], self.w_opt.shape[0]
        )
    )
    return designs @ self.w_opt
