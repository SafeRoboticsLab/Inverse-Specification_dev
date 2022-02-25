import os
import time
import numpy as np
from swri.flight_dynamics import SWRIFlightDynamics, SWRIFlightDynamicsParallel
from swri.problem import SWRIElementwiseProblem, SWRIProblem
from utils import sample_and_evaluate, set_seed

# template_file: the path to the architecture of the aircraft
# exec_file: the path to the flight dynamics model
TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
EXEC_FILE = os.path.join('swri', "new_fdm")
speed_list = (np.arange(10) + 1) * 5
test_simulator = False
test_problem_wrapper = False

# test simulator
if test_simulator:
  X = []
  for speed in speed_list:
    X.append(
        dict(
            control_iaileron=5,
            control_iflap=6,
            control_i_flight_path=5,  # Fix to one of these 1, 3, 4, 5
            control_Q_position=3.9971661079507594,
            control_Q_velocity=3.6711272495701843,
            control_Q_angular_velocity=3.3501992857774856,
            control_Q_angles=3.0389318577493087,
            control_R=4.422413267471787,
            control_requested_lateral_speed=speed,
            control_requested_vertical_speed=0,
        )
    )

  simulator = SWRIFlightDynamics(TEMPLATE_FILE, EXEC_FILE)
  exec_time_series = 0.
  scores_series = np.empty_like(speed_list)

  for i, x in enumerate(X):
    start_time = time.time()
    y = simulator.sim(x, delete_folder=True)
    exec_time_series += (time.time() - start_time)
    scores_series[i] = y['Path_traverse_score_based_on_requirements']
    if i == 2:
      print("Use control_requested_lateral_speed: {}".format(speed_list[i]))
      print("\nGet the output from the simulator:")
      for key, value in y.items():
        print(key, ":", value)

  print("\n== Serial Computation ==")
  print("scores:", scores_series)
  print("--> EXEC TIME: {}".format(exec_time_series))

  simulator_parallel = SWRIFlightDynamicsParallel(
      TEMPLATE_FILE, EXEC_FILE, num_workers=5
  )
  exec_time_parallel = 0.
  start_time = time.time()
  Y = simulator_parallel.sim(X, delete_folder=False)
  exec_time_parallel += (time.time() - start_time)
  scores_parallel = np.empty_like(speed_list)
  for i, y in enumerate(Y):
    scores_parallel[i] = y['Path_traverse_score_based_on_requirements']

  print("\n== Parallel Computation ==")
  print("scores:", scores_parallel)
  print("--> EXEC TIME: {}".format(exec_time_parallel))

# test problem wrapper
if test_problem_wrapper:
  X = np.empty(shape=(len(speed_list), 6))
  X[:, 0] = 3.9971661079507594
  X[:, 1] = 3.6711272495701843
  X[:, 2] = 3.3501992857774856
  X[:, 3] = 3.0389318577493087
  X[:, 4] = 4.422413267471787
  X[:, 5] = speed_list
  print("\n")
  problem = SWRIElementwiseProblem(TEMPLATE_FILE, EXEC_FILE)
  print(problem.objective_names.values())
  scores_series = np.empty_like(speed_list)
  exec_time_series = 0.
  for i, x in enumerate(X):
    start_time = time.time()
    y = {}
    problem._evaluate(x, y, get_score=True)
    exec_time_series += (time.time() - start_time)
    scores_series[i] = y['F']
    if i == 2:
      tmp = {}
      problem._evaluate(x, tmp)
      print(tmp['F'])
  print("\n== Serial Problem Simulator ==")
  print("scores:", scores_series)
  print("--> EXEC TIME: {}".format(exec_time_series))

  problem_parallel = SWRIProblem(TEMPLATE_FILE, EXEC_FILE, 5)
  out_parallel = {}
  exec_time_parallel = 0.
  start_time = time.time()
  problem_parallel._evaluate(X, out_parallel)
  exec_time_parallel += (time.time() - start_time)
  scores_parallel = out_parallel['scores']
  print("\n== Parallel Problem Simulator ==")
  print("scores:", scores_parallel)
  print("--> EXEC TIME: {}".format(exec_time_parallel))

  out_tmp = {}
  problem_parallel._evaluate(
      X[2:3], out_tmp, get_score=False, delete_folder=False
  )
  print(out_tmp["F"])

if True:
  # X = np.empty(shape=(len(speed_list), 6))
  # X[:, 0] = 3.9971661079507594
  # X[:, 1] = 3.6711272495701843
  # X[:, 2] = 3.3501992857774856
  # X[:, 3] = 3.0389318577493087
  # X[:, 4] = 4.422413267471787
  # X[:, 5] = speed_list
  set_seed(seed_val=0, use_torch=False)

  problem = SWRIProblem(TEMPLATE_FILE, EXEC_FILE, 5)
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  components, y = sample_and_evaluate(problem, component_values_bound, 5)

  with np.printoptions(formatter={'float': '{: .4e}'.format}):
    print("components:")
    print(components)
    # print(y["scores"])

    print("\nfeatures:")
    print(y["F"])
