# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC
from typing import List, Union
from .ranker.ranker import Ranker
from .confirmer.empty_confirmer import EmptyConfirmer
from invspec.design import Design


class HumanSimulator(ABC):

  def __init__(self, ranker: Ranker, confirmer=EmptyConfirmer()):
    super().__init__()

    self.ranker = ranker
    self.confirmer = confirmer
    self.num_ranking_queries = 0
    self.num_confirmation_queries = 0

  def get_ranking(self, query: List[Design], **kwargs) -> Union[int, List]:
    """Gets the preference of designs or returns "cannot distinguish".

    Args:
        query (List[Design]).
    """
    self.num_ranking_queries += 1
    return self.ranker.get_ranking(query, **kwargs)

  def confirm(self, query: Design, **kwargs):
    """Accepts or rejects this design.

    Args:
        query (Design).
    """
    self.num_confirmation_queries += 1
    return self.confirmer.confirm(query, **kwargs)

  def get_num_ranking_queries(self) -> int:
    return self.num_ranking_queries

  def get_num_confirmation_queries(self) -> int:
    return self.num_confirmation_queries
