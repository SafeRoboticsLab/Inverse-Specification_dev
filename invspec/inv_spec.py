# from abc import abstractmethod
import numpy as np

from invspec.querySelector.random_selector import RandomQuerySelector


class InvSpec(object):

  def __init__(self, inference, querySelector=RandomQuerySelector(), **kwargs):
    """

    Args:
        inference (:class: `Inference`): specific inference method used to
            provide human fitness.
        querySelector (:class: `QuerySelector`): specific query selection
            method. Defaults to RandomQuerySelector().
    """

    super().__init__()

    self.inference = inference
    self.querySelector = querySelector

  #== INTERACT WITH GA AND HUMAN ==
  def evaluate(self, pop, **kwargs):
    return self.inference.eval(pop, **kwargs)

  def get_query(self, pop, n_queries, n_designs=2, **kwargs):
    return self.querySelector.do(
        pop, n_queries, n_designs, evalFunc=self.inference.eval_query, **kwargs
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
