# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from humansim.confirmer.confirmer import Confirmer


class EmptyConfirmer(Confirmer):

  def __init__(self):
    super().__init__()

  def _confirm(self, query, **kwargs):
    return True
