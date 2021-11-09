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
        query (dict):
            'F' (np.ndarray, (#designs x #features)): designs represented by
                their features (objectives defined in `problem`).
            'X' (np.ndarray, (#designs x #components)): designs represented by
                their component values (inputs defined in `problem`).
    """
    return self.ranker.get_ranking(query, **kwargs)

  def confirm(self, query, **kwargs):
    """Accepts or rejects this design.

    Args:
        query (dict):
            'F' (np.ndarray, (#designs x #features)): designs represented by
                their features (objectives defined in `problem`).
            'X' (np.ndarray, (#designs x #components)): designs represented by
                their component values (inputs defined in `problem`).
    """
    return self.confirmer.confirm(query, **kwargs)
