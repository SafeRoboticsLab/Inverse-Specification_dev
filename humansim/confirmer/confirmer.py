# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from abc import ABC, abstractmethod
from invspec.design import Design


class Confirmer(ABC):

  def __init__(self):
    super().__init__()

  def confirm(self, query: Design, **kwargs) -> bool:
    """Accepts or rejects this design.

    Args:
        query (np.array, (#features,)): a design represented by its features
            (objectives defined in `problem`)
    """
    return self._confirm(query, **kwargs)

  @abstractmethod
  def _confirm(self, query: Design, **kwargs) -> bool:
    raise NotImplementedError
