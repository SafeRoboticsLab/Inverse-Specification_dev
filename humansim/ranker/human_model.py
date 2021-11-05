# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np
import os


os.sys.path.append('..')
from utils import findInfeasibleDesigns


#== Human Feedback ==
def HumanFeedback(
    designFeature, w_opt, considerConstarint=False, activeConstraintSet=None
):
  """
  Gets the human's preference order.

  Args:
      designFeature (np.ndarray): float, designs' features.
      w_opt (np.ndarray): float, optimal weight.
      considerConstarint (bool, optional): considering constraints or not.
          Defaults to False.
      activeConstraintSet (list, optional): active constraints. Defaults to
          None.

  Returns:
      order (np.ndarray): int, human's preference, from high to low.
  """
  if considerConstarint:
    indicator = findInfeasibleDesigns(designFeature, activeConstraintSet)
    infeasibleIndex = np.arange(designFeature.shape[0])[indicator]
    feasibleIndex = np.arange(designFeature.shape[0]
                             )[np.logical_not(indicator)]

    feasibleDesigns = designFeature[feasibleIndex]
    scores = feasibleDesigns @ w_opt
    orderFirstPart = feasibleIndex[np.argsort(-scores)]

    order = np.concatenate((orderFirstPart, infeasibleIndex))
    infeasibleIndicatorSorted = indicator[order]
  else:
    infeasibleIndicatorSorted = np.full(
        shape=(designFeature.shape[0],), fill_value=False
    )
    scores = designFeature @ w_opt
    order = np.argsort(-scores)

  return order, infeasibleIndicatorSorted


def HumanFeedbackPair(
    designFeature, w_opt, beta=5., considerConstarint=False,
    activeConstraintSet=None, perfectRank=False
):
  if considerConstarint:
    indicator = findInfeasibleDesigns(designFeature, activeConstraintSet)
    # infeasibleIndex = np.arange(designFeature.shape[0])[indicator]
    feasibleIndex = np.arange(designFeature.shape[0]
                             )[np.logical_not(indicator)]
    feasibleDesigns = designFeature[feasibleIndex]
    scores = feasibleDesigns @ w_opt
    orderFirstPart = feasibleIndex[np.argsort(-scores)]

    if feasibleDesigns.shape[0] == 0:
      feedback = 2
    elif feasibleDesigns.shape[0] == 1:
      feedback = feasibleIndex[0]
    else:
      p = 1 / (1 + np.exp(beta * (scores[0] - scores[1])))
      if perfectRank:
        feedback = orderFirstPart[0]
      else:
        if p < 0.6 and p > 0.4:
          feedback = 2
        else:
          feedback = orderFirstPart[0]

    # if feasibleDesigns.shape[0] == 1:
    #     feedback = feasibleIndex[0]
    # else:
    #     scores = designFeature @ w_opt
    #     order = np.argsort(-scores)
    #     feedback = order[0]
  else:
    scores = designFeature @ w_opt
    order = np.argsort(-scores)

    p = 1 / (1 + np.exp(beta * (scores[0] - scores[1])))
    if p < 0.6 and p > 0.4:
      feedback = 2
    else:
      feedback = order[0]

  return feedback


#== Likelihood ==
def HumanSelectionModel(order, w, designFeature, beta):
  """Gets the likelihood of the preference given the weight and designs.

  Args:
      order (np.ndarray): int, human's preference order.
      w (np.ndarray): float, a weight.
      designFeature (np.ndarray): float, designs' features.
      beta (float): confidence coefficient.

  Returns:
      P (float): likelihood
  """
  P = 1
  utility = beta * designFeature @ w.reshape(-1, 1)

  for k in range(order.shape[0] - 1):
    denominator = np.sum(np.exp(utility[order[k:]]))
    numerator = np.exp(utility[order[k]])
    P_k = numerator / denominator
    P *= P_k

  return P


def HumanSelectionModelByPair(order, w, designFeature, beta):
  """Gets the likelihood given the order of a design pair.

  Args:
      order (np.ndarray): int, human's preference order.
      w (np.ndarray): float, a weight.
      designFeature (np.ndarray): float, designs' features.
      beta (float): confidence coefficient.

  Returns:
      (float): likelihood ratio
  """
  assert designFeature.shape[
      0] == 2, "This function is only used for comparing design pairs!!!"

  utility = beta * designFeature @ w.reshape(-1, 1)

  return 1 / (1 + np.exp(utility[order[1]] - utility[order[0]]))
