# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from typing import Union, List
import numpy as np

from invspec.query_selector.base_query_selector import QuerySelector
from invspec.design import Design


class RandomQuerySelector(QuerySelector):

  def __init__(self):
    super().__init__()

  def _do(
      self, pop: Union[List[Design], np.ndarray], n_queries: int,
      n_designs: int, **kwargs
  ) -> np.ndarray:
    assert n_designs == 2, "Only implemented for pairs of designs!"
    n_pop = len(pop)
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
