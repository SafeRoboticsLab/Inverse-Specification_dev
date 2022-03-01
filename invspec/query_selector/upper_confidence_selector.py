# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from typing import Union, Callable
import numpy as np
from pymoo.core.population import Population

from invspec.query_selector.base_query_selector import QuerySelector


class UCBQuerySelector(QuerySelector):

  def __init__(self, tradeoff=1.):
    super().__init__()
    self.tradeoff = tradeoff

  def _do(
      self,
      pop: Union[Population, np.ndarray],
      n_queries: int,
      n_designs: int,
      eval_func: Callable[[Union[Population, np.ndarray], float], np.ndarray],
      update_times: int,
      **kwargs,
  ) -> np.ndarray:
    """
    Picks the designs having the highest upper confidence bound from the
    current population.

    Args:
        pop (pymoo.core.population.Population | numpy.ndarray): The population
            which should be selected from.
        n_queries (int): Number of queries to send.
        n_designs (int): Number of designs in each query.
        eval_func (Callable): the function for evaluating designs, which
            returns the upper confidence bound.
        update_times (int): Number of updates of inference engine.

    Returns:
        ndarray: Indices of selected individuals, (n_queries, 2).
    """
    assert n_designs == 2, "Only implemented for pairs of designs!"
    n_pop = len(pop)
    pool_size = kwargs.get('pool_size', n_queries)
    add_cur_best = kwargs.get('add_cur_best', False)

    if update_times > 0:
      print("Select queries based on UCB!")
      ucb = eval_func(pop, self.tradeoff)
      # randomized argsort, np method is ascending
      P = np.random.permutation(n_pop)
      indices = np.argsort(-ucb[P])
      indices = P[indices[:pool_size]]
    else:
      indices = np.arange(n_pop)

    # samples from the pool randomly.
    if add_cur_best:  # append a placeholder for the current best design
      indices = np.append(indices, -1)
    n_pool = len(indices)

    query_selected_mtx = np.full((n_pool, n_pool), False, dtype=bool)
    I = np.empty((n_queries, n_designs), dtype=int)
    idx = 0

    while idx < n_queries:
      idx2cand = np.random.choice(n_pool, n_designs, replace=False)
      if not query_selected_mtx[idx2cand[0], idx2cand[1]]:
        query_cand = (indices[idx2cand[0]], indices[idx2cand[1]])
        I[idx, :] = query_cand
        query_selected_mtx[idx2cand[0], idx2cand[1]] = True
        query_selected_mtx[idx2cand[1], idx2cand[0]] = True
        idx += 1

    return I
