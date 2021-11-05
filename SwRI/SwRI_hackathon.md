# DesignSpace, Name=SwRI
## Input
<!-- battery_capacity, input, continuous, 1100.0 -->
* control_i_flight_path, input, discrete, {1, 3, 4, 5}
* control_Q_position, input, continuous, 1.
* control_Q_velocity, input, continuous, 1.
* control_Q_angular_velocity, input, continuous, 0.
* control_Q_angles, input, continuous, 1.
* control_R, input, continuous, 1.
* control_requested_lateral_speed, input, continuous, 0.0
* control_requested_vertical_speed, input, continuous, -1.0

## Bounds
<!-- propeller_radius, 104, 173 -->
<!-- battery_capacity, 1100.0, 6000.0 ?? -->
<!-- control_Q_angles, 0.0, 5.0 -->
<!-- control_i_flight_path, 1, 1 -->
* control_Q_angular_velocity, 0.0, 5.0
* control_Q_position, 0.0, 5.0
* control_Q_velocity, 0.0, 5.0
* control_Q_angles, 0.0, 5.0
* control_R, 0.0, 5.0
* control_requested_vertical_speed, 0.0, 0.0
* control_requested_lateral_speed, 0.0, 50.0

## Output
<!-- * Flight_distance, output, continuous
* Time_to_traverse_path, output, continuous
* Time_of_maximum_distance_error, output, continuous
* Maximimum_error_distance_during_flight, output, continuous
* Path_traverse_score_based_on_requirements, output, continuous -->
* Max_Hover_Time_s
* Max_Lateral_Speed_ms
* Max_Flight_Distance_m
* Speed_at_Max_Flight_Distance_ms
* Max_uc_at_Max_Flight_Distance
* Power_at_Max_Flight_Distance_W
* Motor_amps_to_max_amps_ratio_at_Max_Flight_Distance
* Motor_power_to_max_power_ratio_at_Max_Flight_Distance
* Battery_amps_to_max_amps_ratio_at_Max_Flight_Distance
* Distance_at_Max_Speed_m
* Power_at_Max_Speed_W
* Motor_amps_to_max_amps_ratio_at_Max_Speed
* Motor_power_to_max_power_ratio_at_Max_Speed
* Battery_amps_to_max_amps_ratio_at_Max_Speed
* Hackathon
* Flight_distance
* Time_to_traverse_path
* Average_speed_to_traverse_path
* Maximimum_error_distance_during_flight
* Time_of_maximum_distance_error
* Spatial_average_distance_error
* Maximum_ground_impact_speed
* Path_traverse_score_based_on_requirements

## Simulator
* SWRIFlightDynamics with kwargs
  * template_file
  * exec_file
* The black-box simulator can be obtained here:
https://git.isis.vanderbilt.edu/SwRI/flight-dynamics-model

## Optimize
### Variables
* control_Q_position
* control_Q_velocity
* control_Q_angular_velocity
* control_Q_angles
* control_R
* control_requested_lateral_speed
* control_requested_vertical_speed

### Objectives
* battery_capacity, min ??
* Path_traverse_score_based_on_requirements, max
