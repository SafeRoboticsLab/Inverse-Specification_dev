# from abc import abstractmethod
import numpy as np

from invspec.query_selector.random_selector import RandomQuerySelector
from invspec.inference.inference import Inference


class InvSpec(object):

  def __init__(
      self, inference: Inference, query_selector=RandomQuerySelector(),
      **kwargs
  ):
    """

    Args:
        inference (:class: `Inference`): specific inference method used to
            provide human fitness.
        query_selector (:class: `query_selector`): specific query selection
            method. Defaults to RandomQuerySelector().
    """

    super().__init__()

    self.inference = inference
    self.query_selector = query_selector

  #== INTERACT WITH GA AND HUMAN ==
  def evaluate(self, pop, **kwargs):
    return self.inference.eval(pop, **kwargs)

  def get_query(
      self, pop, n_queries, n_designs=2, eval_func=None, **kwargs
  ) -> np.ndarray:
    if eval_func is None:
      eval_func = self.inference.eval_query
    return self.query_selector.do(
        pop, n_queries, n_designs, eval_func=eval_func,
        update_times=self.inference.update_times, **kwargs
    )

  def normalize(self, F):
    return self.inference.normalize(F)

  #== INFERENCE RELATED FUNCTIONS ==
  def clear_feedback(self):
    self.inference.clear_feedback()

  def store_feedback(self, *args):
    self.inference.store_feedback(*args)

  def get_sampled_query_feedback(self, size):
    return self.inference.get_sampled_query_feedback(size)

  def get_number_feedback(self):
    return len(self.inference.memory)

  def initialize(self, *args, **kargs):
    self.inference.initialize(*args, **kargs)

  def learn(self, *args, **kargs):
    return self.inference.learn(*args, **kargs)
