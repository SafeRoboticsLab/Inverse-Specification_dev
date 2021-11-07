import os
import numpy as np
from flight_dynamics import SWRIFlightDynamics
from problem import SWRIsim

# template_file: the path to the architecture of the aircraft
# exec_file: the path to the flight dynamics model
head_dir = os.path.dirname(os.path.realpath(__file__))
TEMPLATE_FILE = os.path.join(head_dir, 'template', 'FlightDyn_quadH.inp')
EXEC_FILE = os.path.join(head_dir, "new_fdm")

simulator = SWRIFlightDynamics(TEMPLATE_FILE, EXEC_FILE)

x = dict(
    control_iaileron=5,
    control_iflap=6,
    control_i_flight_path=5,  # Fix to one of these 1, 3, 4, 5
    control_Q_position=3.9971661079507594,
    control_Q_velocity=3.6711272495701843,
    control_Q_angular_velocity=3.3501992857774856,
    control_Q_angles=3.0389318577493087,
    control_R=4.422413267471787,
    control_requested_lateral_speed=26.019805936722737,
    control_requested_vertical_speed=0,
)
y = simulator.sim(x, delete_folder=False)
print("\nGet the output from the simulator:")
for key, value in y.items():
  print(key, ":", value)

problem = SWRIsim(TEMPLATE_FILE, EXEC_FILE)
x = np.array([
    3.9971661079507594, 3.6711272495701843, 3.3501992857774856,
    3.0389318577493087, 4.422413267471787, 26.019805936722737
])
y = {}
problem._evaluate(x, y)
print("\nGet the output from the problem:")
for key, value in y.items():
  print(key, ":", value)
