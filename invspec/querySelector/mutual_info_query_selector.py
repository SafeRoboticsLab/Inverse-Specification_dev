# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Active selection is modified from:
# https://github.com/Stanford-ILIAD/active-preference-based-gpr

import numpy as np

from invspec.querySelector.base_query_selector import QuerySelector


class MutualInfoQuerySelector(QuerySelector):

  def __init__(self):
    super().__init__()

  def _do(self, pop, n_queries, n_designs, **kwargs):
    """
    pick the most informative pairs of designs out of the current population,
    the metric is based on mutual information (a.k.a information gain).

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
    assert n_designs == 2, "Only implemented for pairs of designs!"
    evalFunc = kwargs.get('evalFunc')
    n_pop = len(pop)

    if self.queryTimes > 0:
      # IG_mtx = np.zeros(shape=(n_pop, n_pop))
      n_values = int(n_pop * (n_pop-1) / 2)
      IG = np.zeros(shape=(n_values,))
      _I = np.empty((n_values, n_designs), dtype=int)
      idx = 0
      for i in range(n_pop):
        for j in range(i + 1, n_pop):
          IG[idx] = evalFunc(pop[[i, j]])
          _I[idx] = i, j
          idx += 1

      # randomized argsort
      P = np.random.permutation(n_values)
      indices = np.argsort(-IG[P])
      indices = P[indices]
      # indices = np.argsort(-IG)
      I = _I[indices[:n_queries]]
    else:
      queryMtx = np.full((n_pop, n_pop), False, dtype=bool)
      I = np.empty((n_queries, n_designs), dtype=int)
      idx = 0

      while idx < n_queries:
        # print(idx, end='\r')
        optPair = np.random.choice(n_pop, n_designs, replace=False)
        if not queryMtx[optPair[0], optPair[1]]:
          I[idx, :] = optPair
          queryMtx[optPair[0], optPair[1]] = True
          queryMtx[optPair[1], optPair[0]] = True
          idx += 1

    return I
