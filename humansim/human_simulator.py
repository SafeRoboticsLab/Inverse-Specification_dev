# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC
from .confirmer.empty_confirmer import EmptyConfirmer


class HumanSimulator(ABC):

  def __init__(self, ranker, confirmer=EmptyConfirmer()):
    super().__init__()

    self.ranker = ranker
    self.confirmer = confirmer

  def get_ranking(self, query, **kwargs):
    """Gets the preference of designs or returns "cannot distinguish".

    Args:
        query (np.array, (#designs x #features)): designs represented by
            their features (objectives defined in `problem`)
    """
    return self.ranker.get_ranking(query, **kwargs)

  def confirm(self, query, **kwargs):
    """Accepts or rejects this design.

    Args:
        query (np.array, (#features,)): a design represented by its features
            (objectives defined in `problem`)
    """
    return self.confirmer.confirm(query, **kwargs)
