# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np

from invspec.querySelector.base_query_selector import QuerySelector


class RandomQuerySelector(QuerySelector):

  def __init__(self):
    super().__init__()

  def _do(self, pop, n_queries, n_designs, **kwargs):
    assert n_designs == 2, "Only implemented for pairs of designs!"
    n_pop = len(pop)
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
