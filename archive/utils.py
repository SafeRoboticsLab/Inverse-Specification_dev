import numpy as np


def init_experiment(num_weight=50, dim=3, use_one_norm=False, shrink=.95):
  """
  Samples weights from 2-norm or 1-norm ball. Generates designs given the
  weights.

  Args:
      num_weight (int, optional):  the number of weights. Defaults to 50.
      dim (int, optional):        the dimension of a weight. Defaults to 3.
      use_one_norm (bool, optional): project weights to 1-norm ball. Defaults
          to False.
      shrink (float, optional):   the ratio to shrink designs' features except
          the optimal design with respect to the optimal weight. Defaults to
          0.95.

  Returns:
      weight_array (float array):      array of weights.
      w_opt (float array):            optimal weight.
      idx_opt (int):                  index of the optimal weight.
      design_feature (float array):    array of designs' features
  """
  #= Weight Space
  weight_array, idx_opt = set_weight_space(
      num_weight=num_weight, dim=dim, use_one_norm=use_one_norm
  )
  #= Design Space
  # num_design = weight_array.shape[0]
  design_feature = set_design_space(weight_array, idx_opt, shrink=shrink)

  if use_one_norm:  # project to ||w||_1 = 1
    L1Norm = np.linalg.norm(weight_array, axis=1, ord=1).reshape(-1, 1)
    weight_array /= L1Norm
    #design_feature *= L1Norm
  w_opt = weight_array[idx_opt].copy()

  return weight_array, w_opt, idx_opt, design_feature


def set_weight_space(num_weight=50, dim=3, use_one_norm=False):
  """
  Samples weights from 2-norm or 1-norm ball.

  Args:
      num_weight (int, optional): the number of weights. Defaults to 50.
      dim (int, optional): the dimension of a weight. Defaults to 3.
      use_one_norm (bool, optional): project weights to 1-norm ball. Defaults
          to False.

  Returns:
      weight_array (float array): array of weights.
      w_opt (float array): optimal weight.
      idx_opt (int): index of the optimal weight.
  """
  # ref: https://www.sciencedirect.com/science/article/pii/S0047259X10001211

  weight_array = np.random.normal(size=(num_weight, dim))
  weight_array = np.abs(weight_array)
  weight_array /= np.linalg.norm(weight_array, axis=1, ord=2).reshape(-1, 1)
  # print(
  #     "The shape of the weight array is {:d} x {:d}.".format(
  #         weight_array.shape[0], weight_array.shape[1]
  #     )
  # )

  idx_opt = np.random.choice(num_weight)

  return weight_array, idx_opt


def set_design_space(weight_array, idx_opt, shrink=0.95):
  """
  Generates designs given the weights.

  Args:
      weight_array (float array): array of weights.
      idx_opt (int): index of the optimal weight.
      shrink (float, optional): the ratio to shrink designs' features except
          the optimal design with respect to the optimal weight. Defaults to
          0.95.

  Returns:
      design_feature (float array): array of designs' features
  """
  num_weight = weight_array.shape[0]
  design_feature = weight_array.copy()
  design_feature[np.arange(num_weight) != idx_opt, :] *= shrink

  return design_feature
