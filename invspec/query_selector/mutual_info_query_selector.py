# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Active selection is modified from:
# https://github.com/Stanford-ILIAD/active-preference-based-gpr

from typing import Union
import numpy as np
from pymoo.core.population import Population

from invspec.query_selector.base_query_selector import QuerySelector


class MutualInfoQuerySelector(QuerySelector):

  def __init__(self):
    super().__init__()

  def _do(
      self, pop: Union[Population, np.ndarray], n_queries: int, n_designs: int,
      **kwargs
  ) -> np.ndarray:
    """
    Picks the most informative pairs of designs out of the current population,
    the metric is based on mutual information (a.k.a information gain).

    Args:
        pop (pymoo.core.population.Population | numpy.ndarray): The population
            which should be selected from.
        n_queries (int): Number of queries to send.
        n_designs (int): Number of designs in each query.

    Returns:
        ndarray: Indices of selected individuals.
    """
    assert n_designs == 2, "Only implemented for pairs of designs!"
    eval_func = kwargs.get('eval_func')
    n_pop = len(pop)

    if self.num_query_times > 0:
      # IG_mtx = np.zeros(shape=(n_pop, n_pop))
      n_values = int(n_pop * (n_pop-1) / 2)
      IG = np.zeros(shape=(n_values,))
      _I = np.empty((n_values, n_designs), dtype=int)
      idx = 0
      for i in range(n_pop):
        for j in range(i + 1, n_pop):
          IG[idx] = eval_func(pop[[i, j]])
          _I[idx] = i, j
          idx += 1

      # randomized argsort
      P = np.random.permutation(n_values)
      indices = np.argsort(-IG[P])
      indices = P[indices]
      # indices = np.argsort(-IG)
      I = _I[indices[:n_queries]]
    else:
      query_selected_mtx = np.full((n_pop, n_pop), False, dtype=bool)
      I = np.empty((n_queries, n_designs), dtype=int)
      idx = 0

      while idx < n_queries:
        # print(idx, end='\r')
        query_cand = np.random.choice(n_pop, n_designs, replace=False)
        if not query_selected_mtx[query_cand[0], query_cand[1]]:
          I[idx, :] = query_cand
          query_selected_mtx[query_cand[0], query_cand[1]] = True
          query_selected_mtx[query_cand[1], query_cand[0]] = True
          idx += 1

    return I
