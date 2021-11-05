# Problem = AUVsim_p1
This design space has 4 Discrete/integer input parameters, 9 continuous intput paramters, 6 output parameters.
* Simulator: see `problem.py` and `AUV_sim.py`
* GA solver: see `spreadsheetPymoo.py` and `pymoo` pkg

## Inputs
* `<name>, input, <type>, <example>`
* Discrete and Integer Variables
    * Diameter, input, discrete, 0
    * PV_Material_Choice, input, discrete, 0
    * Battery_Specific_Energy, input, discrete, 0
    * Design_Hotel_Power_Draw, input, discrete, 0
* Continuous Variables
    * Length, input, continuous, 2.2
    * Depth_Rating, input, continuous, 300.
    * Safety_Factor_for_Depth_Rating, input, continuous, 1.2
    * Battery_Fraction, input, continuous, 0.5
    * Cd_Drag_coefficient, input, continuous, 0.0079
    * Appendage_added_area, input, continuous, 0.1
    * Propulsion_efficiency, input, continuous, 0.5
    * Density_of_seawater, input, continuous, 1027.
    * CruiseSpeed, input, continuous, 0.85

## Outputs
* Constraints
    * c1, 0.1 < PV Length, output, continuous
    * c2, 0. < Battery weight, output, continuous
    * c3, c4, 5.5 < Length/Diameter < 7.5, output, continuous
* Objectives
    * Range_at_Efficient_Speed, output, continuous
    * Efficient_Speed, output, continuous

## Bounds
* Discrete variables
    * Diameter, [0.25:0.05:0.45, 0.6, 0.75, 0.8, 1.0]
    * PV_Material_Choice, [1:1:5]
    * Battery_Specific_Energy, [100., 125., 200., 250., 360.]
    * Design_Hotel_Power_Draw, [20.:0.25:22., 40:1:45]
* Continuous variables bounds are defined as LB, UB
    * Length, 1., 5.
    * Depth_Rating, 200., 500.
    * Safety_Factor_for_Depth_Rating, 1.33, 2.
    * Battery_Fraction, 0.4, 0.6
    * Cd_Drag_coefficient, 0.0078, 0.0080
    * Appendage_added_area, 0.1, 0.2
    * Propulsion_efficiency, 0.45, 0.55
    * Density_of_seawater, 1025., 1030.
    * CruiseSpeed, 0.5, 1.5

## Tool
NSGA-II, 100 population size, 100 offspring size, 200 generations