import os
from flight_dynamics import SWRIFlightDynamics

# template_file: the path to the architecture of the aircraft
# exec_file: the path to the flight dynamics model
sim_dict = {
    # 'template_file': 'FlightDyn_7By3.inp',
    'template_file': os.path.join('template', 'FlightDyn_quadH.inp'),
    # 'exec_file': os.path.join("flight-dynamics-model", "bin", "new_fdm")
}

simulator = SWRIFlightDynamics(**sim_dict)

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
y = simulator.sim(x)
print("\nGet the output from the simulator:")
for key, value in y.items():
  print(key, ":", value)
