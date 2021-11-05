# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np

from humansim.ranker.ranker import Ranker
from utils import findInfeasibleDesigns


class PairRanker(Ranker):

  def __init__(
      self, w_opt, beta=5., activeConstraintSet=None, perfectRank=False,
      indifference=0.1
  ):

    super().__init__()
    self.w_opt = w_opt
    self.beta = beta
    self.considerConstarint = (activeConstraintSet is not None)
    self.activeConstraintSet = activeConstraintSet
    self.perfectRank = perfectRank
    self.indifference = indifference

  def _getRanking(self, query, **kwargs):
    n_designs, n_features = query.shape
    assert n_designs == 2, "This ranker only supports binary preference!"
    assert n_features == self.w_opt.shape[0], \
        "#features ({}) doesn't match #weights ({}).".format(
            n_features, self.w_opt.shape[0])

    if self.considerConstarint:
      indicator = findInfeasibleDesigns(query, self.activeConstraintSet)
      # infeasibleIndex = np.arange(query.shape[0])[indicator]
      feasibleIndex = np.arange(n_designs)[np.logical_not(indicator)]
      feasibleDesigns = query[feasibleIndex]
      scores = feasibleDesigns @ self.w_opt
      orderFirstPart = feasibleIndex[np.argsort(-scores)]

      if feasibleDesigns.shape[0] == 0:
        feedback = 2
      elif feasibleDesigns.shape[0] == 1:
        feedback = feasibleIndex[0]
      else:
        p = 1 / (1 + np.exp(self.beta * (scores[0] - scores[1])))
        if self.perfectRank:
          feedback = orderFirstPart[0]
        else:
          if np.abs(p - 0.5) < self.indifference:
            feedback = 2
          else:
            feedback = orderFirstPart[0]
    else:
      scores = query @ self.w_opt
      order = np.argsort(-scores)

      p = 1 / (1 + np.exp(self.beta * (scores[0] - scores[1])))
      if self.perfectRank:
        feedback = order[0]
      else:
        if np.abs(p - 0.5) < self.indifference:
          feedback = 2
        else:
          feedback = order[0]

    return feedback
