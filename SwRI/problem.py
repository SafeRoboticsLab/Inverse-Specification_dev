# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np

from flight_dynamics import SWRIFlightDynamics
from pymoo.core.problem import ElementwiseProblem


class SWRIsim(ElementwiseProblem):

  def __init__(self, template_file, exec_file):
    self.sim = SWRIFlightDynamics(template_file, exec_file)
    xl = np.zeros(6)
    xu = np.array([5., 5., 5., 5., 5., 50.])

    super().__init__(n_var=6, n_obj=7, n_constr=0, xl=xl, xu=xu)

  def _input_wrapper(self, x):
    """Wraps designs to fit in the format of the simulator input.

    Args:
        x (np.ndarray): features of a design

    Returns:
        dict: an input for the simulator.
    """
    input = dict(
        # region: fixed parameters
        control_iaileron=5,
        control_iflap=6,
        control_i_flight_path=5,
        control_requested_vertical_speed=0,
        # endregion
        # region: inputs, LQR parameters
        control_Q_position=x[0],
        control_Q_velocity=x[1],
        control_Q_angular_velocity=x[2],
        control_Q_angles=x[3],
        control_R=x[4],
        control_requested_lateral_speed=x[5],
        # endregion
    )
    return input

  def _output_extracter(self, output, get_score=False, **kwargs):
    y = []
    for key, value in output.items():
      # if (
      #     key != "Hackathon"
      #     and key != "Path_traverse_score_based_on_requirements"
      # ):
      if ((key == "Flight_distance") or (key == "Time_to_traverse_path")
          or (key == "Average_speed_to_traverse_path")
          or (key == "Maximimum_error_distance_during_flight")
          or (key == "Time_of_maximum_distance_error")
          or (key == "Spatial_average_distance_error")
          or (key == "Maximum_ground_impact_speed")):
        y.append(value)

    return np.array(y)

  def _evaluate(self, x, out, *args, **kwargs):
    input = self._input_wrapper(x)
    output = self.sim.sim(input, delete_folder=True)
    out["F"] = self._output_extracter(output, get_score=False)
