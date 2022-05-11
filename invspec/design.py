# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from __future__ import annotations
from typing import Any, Mapping, Tuple, Union, List
import numpy as np
import copy


def design2metrics(designs: List[Design], key: int) -> np.ndarray:
  metrics = np.array([design.get_test_metrics(key) for design in designs],
                     dtype=np.float32)

  return metrics


def design2params(designs: List[Design], key: int) -> np.ndarray:
  params = np.array([design.get_test_params(key) for design in designs],
                    dtype=np.float32)

  return params


class Design():
  catalog = set()

  def __init__(
      self, physical_components: dict, test_params: dict[str, np.ndarray],
      test_results: Mapping[str, Mapping[str, np.ndarray]],
      global_features: dict[str, np.ndarray], design_id: Any
  ) -> None:
    self.design_param = DesignParameters(physical_components, test_params)
    self.fwd_features = ForwardFeatures(test_results, global_features)
    self.id = design_id
    self.catalog.add(design_id)

  def __del__(self):
    self.catalog.remove(self.id)

  def get_components(self,
                     key: str) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    return self.design_param.get_components(key)

  def get_test_params(self, key: str) -> np.ndarray:
    return self.design_param.get_test_params(key)

  def get_test_metrics(self, key: str) -> np.ndarray:
    return self.fwd_features.get_test_metrics(key)

  def get_trajectory(self, key: str) -> np.ndarray:
    return self.fwd_features.get_trajectory(key)


class DesignParameters():

  def __init__(
      self, physical_components: dict, test_params: dict[str, np.ndarray]
  ) -> None:
    """

    Args:
        physical_components (dict): key can be "components", "graph", etc.
        test_params (dict): keys can be "test1", "test2", etc. Each value
            should be a numpy ndarray, which is the parameters for that test,
            such as LQR parameters.
    """
    self.physical_components = copy.deepcopy(physical_components)
    self.test_params = copy.deepcopy(test_params)

  def get_components(self,
                     key: str) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    assert key in self.physical_components.keys(), "Key is invalid!"
    return self.physical_components[key]

  def get_test_params(self, key: str) -> np.ndarray:
    assert key in self.test_params.keys(), "Key is invalid!"
    return self.test_params[key]


class ForwardFeatures():

  def __init__(
      self, test_results: Mapping[str, Mapping[str, np.ndarray]],
      global_features: dict[str, np.ndarray]
  ) -> None:
    """

    Args:
        test_results (Mapping[str, Mapping[str, np.ndarray]]): a dictionary
            with keys of "test1", "test2", etc. Each value is also a dictionary
            with keys of "metrics" and "trajectory".
        global_features (dict): keys can be "weight", "dimension", etc.
    """
    self.test_results = test_results
    self.global_features = global_features

  def get_global_features(self, key: str) -> np.ndarray:
    assert key in self.global_features.keys(), "Key is invalid!"
    return self.global_features[key]

  def get_test_metrics(self, key: str) -> np.ndarray:
    assert key in self.test_results.keys(), "Key is invalid!"
    return self.test_results[key]['metrics']

  def get_trajectory(self, key: str) -> np.ndarray:
    assert key in self.test_results.keys(), "Key is invalid!"
    return self.test_results[key]['trajectory']
