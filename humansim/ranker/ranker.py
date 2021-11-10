# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod


class Ranker(ABC):

  def __init__(self):
    super().__init__()

  def get_ranking(self, query, **kwargs):
    """Gets the preference of designs or returns "cannot distinguish".

    Args:
        query (dict):
            'F' (np.ndarray, (#designs x #features)): designs represented by
                their features (objectives defined in `problem`).
            'X' (np.ndarray, (#designs x #components)): designs represented by
                their component values (inputs defined in `problem`).
    """
    return self._get_ranking(query, **kwargs)

  @abstractmethod
  def _get_ranking(self, query, **kwargs):
    raise NotImplementedError
