from pycel import ExcelCompiler
import os


fname = os.path.join(os.path.dirname(__file__), 'AUVRangeSpeedV3.pkl')

# DexcelInterface defines the spread simulator and its inputs and outputs.
# Also, it defines `function()` to access outputs. The objectives and
# constraints are defined in its child class, e.g., DexcelInterface_p1 and
# DexcelInterface_p2. The child class defines `problem()` to access objectives
# and constraints.


class DexcelInterface():

  def __init__(self, filename=fname):
    print("load spreadsheet pickle file from {}".format(fname))
    self.filename = filename
    self.excel = ExcelCompiler.from_file(self.filename)
    self.inputs = {
        'Diameter': 'RangeVsSpeed!B7',
        'Length': 'RangeVsSpeed!B8',
        'PV_Material_Choice': 'RangeVsSpeed!B9',
        'Depth_Rating': 'RangeVsSpeed!B11',
        'Safety_Factor_for_Depth_Rating': 'RangeVsSpeed!B12',
        'Battery_Specific_Energy': 'RangeVsSpeed!B13',
        'Battery_Fraction': 'RangeVsSpeed!B14',
        'Design_Hotel_Power_Draw': 'RangeVsSpeed!B15',
        'Cd_Drag_coefficient': 'RangeVsSpeed!B16',
        'Appendage_added_area': 'RangeVsSpeed!B17',
        'Propulsion_efficiency': 'RangeVsSpeed!B18',
        'Density_of_seawater': 'RangeVsSpeed!B19',
        'CruiseSpeed': 'RangeVsSpeed!B20',
    }
    self.outputs = {
        'PV_Material': 'RangeVsSpeed!B10',
        'Fineness_Ratio': 'RangeVsSpeed!B22',
        'Vehicle_fairing_dispacement': 'RangeVsSpeed!B23',
        'Wetted_Surface_Fairing': 'RangeVsSpeed!B24',
        'Midbody_length': 'RangeVsSpeed!B25',
        'Midbody_volume': 'RangeVsSpeed!B26',
        'PV_Weight': 'RangeVsSpeed!B27',
        'PV_Displacement': 'RangeVsSpeed!B28',
        'Battery_Weight': 'RangeVsSpeed!B29',
        'Battery_Capacity': 'RangeVsSpeed!B30',
        'Nose_Displacement_Wet_Volume': 'RangeVsSpeed!B33',
        'PV_Excess_Buoyancy': 'RangeVsSpeed!B34',
        'Drag': 'RangeVsSpeed!B36',
        'Prop_Power': 'RangeVsSpeed!B37',
        'Range': 'RangeVsSpeed!B38',
        'Efficient_Speed': 'RangeVsSpeed!B39',
        'Range_at_Efficient_Speed': 'RangeVsSpeed!B40',
    }

  def problem(self, x):  # return evaluation of obj. and cons.
    raise NotImplementedError

  def function(self, x):  # return all output values
    for key, value in x.items():
      self.excel.set_value(self.inputs[key], value)
    y = []
    for output in self.outputs.values():
      y.append(self.excel.evaluate(output))
    return y


class DexcelInterface_p1(DexcelInterface):

  def __init__(self, filename=fname):
    super().__init__(filename)
    self.constraints = {
        'c1': 'RangeVsSpeed!F23',
        'c2': '-RangeVsSpeed!F24',
        'c3': 'RangeVsSpeed!F25',
        'c4': 'RangeVsSpeed!F25',
    }
    self.objectives = {
        'o1': 'RangeVsSpeed!B40',
        'o2': 'RangeVsSpeed!B39',
    }
    self.objective_names = {
        'o1': 'Range_at_Efficient_Speed',
        'o2': 'Efficient_Speed'
    }

  def problem(self, x):  # return evaluation of obj. and cons.
    for key, value in x.items():
      self.excel.set_value(self.inputs[key], value)
    y = []
    for objective in self.objectives.values():
      y.append(self.excel.evaluate(objective))
    for name, constraint in self.constraints.items():
      if name == 'c1':
        ev = self.excel.evaluate(constraint)
        y.append(0.1 - ev)
      elif name == 'c3':
        ev = self.excel.evaluate(constraint)
        y.append(5.5 - ev)
      elif name == 'c4':
        ev = self.excel.evaluate(constraint)
        y.append(ev - 7.5)
      elif constraint[0] == '-':
        y.append(-self.excel.evaluate(constraint[1:]))
      else:
        y.append(self.excel.evaluate(constraint))
    return y


class DexcelInterface_p2(DexcelInterface):

  def __init__(self, filename=fname):
    super().__init__(filename)
    self.constraints = {
        'c1': 'RangeVsSpeed!F23',
        'c2': '-RangeVsSpeed!F24',
        'c3': 'RangeVsSpeed!F25',
        'c4': 'RangeVsSpeed!F25',
    }
    self.objectives = {
        'o1': 'RangeVsSpeed!B40',
        'o2': 'RangeVsSpeed!B39',
        'o3': '-RangeVsSpeed!B37',
    }
    self.objective_names = {
        'o1': 'Range_at_Efficient_Speed',
        'o2': 'Efficient_Speed',
        'o3': 'Prop_Power',
    }

  def problem(self, x):  # return evaluation of obj. and cons.
    for key, value in x.items():
      self.excel.set_value(self.inputs[key], value)
    y = []
    for objective in self.objectives.values():
      if objective[0] == '-':
        y.append(-self.excel.evaluate(objective[1:]))
      else:
        y.append(self.excel.evaluate(objective))
    for name, constraint in self.constraints.items():
      if name == 'c1':
        ev = self.excel.evaluate(constraint)
        y.append(0.1 - ev)
      elif name == 'c3':
        ev = self.excel.evaluate(constraint)
        y.append(5.5 - ev)
      elif name == 'c4':
        ev = self.excel.evaluate(constraint)
        y.append(ev - 7.5)
      elif constraint[0] == '-':
        y.append(-self.excel.evaluate(constraint[1:]))
      else:
        y.append(self.excel.evaluate(constraint))
    return y


class DexcelInterface_p3(DexcelInterface):

  def __init__(self, filename=fname):
    super().__init__(filename)
    self.constraints = {
        'c1': 'RangeVsSpeed!F23',
        'c2': '-RangeVsSpeed!F24',
        'c3': 'RangeVsSpeed!F25',
        'c4': 'RangeVsSpeed!F25',
    }
    self.objectives = {
        'o1': 'RangeVsSpeed!B40',
        'o2': 'RangeVsSpeed!B39',
        'o3': 'RangeVsSpeed!B38',
        'o4': 'RangeVsSpeed!B37',
        'o5': 'RangeVsSpeed!B36',
    }
    self.objective_names = {
        'o1': 'Range_at_Efficient_Speed',
        'o2': 'Efficient_Speed',
        'o3': 'Range',
        'o4': 'Prop_Power',
        'o5': 'Drag',
    }

  def problem(self, x):  # return evaluation of obj. and cons.
    for key, value in x.items():
      self.excel.set_value(self.inputs[key], value)
    y = []
    for objective in self.objectives.values():
      y.append(self.excel.evaluate(objective))
    for name, constraint in self.constraints.items():
      if name == 'c1':
        ev = self.excel.evaluate(constraint)
        y.append(0.1 - ev)
      elif name == 'c3':
        ev = self.excel.evaluate(constraint)
        y.append(5.5 - ev)
      elif name == 'c4':
        ev = self.excel.evaluate(constraint)
        y.append(ev - 7.5)
      elif constraint[0] == '-':
        y.append(-self.excel.evaluate(constraint[1:]))
      else:
        y.append(self.excel.evaluate(constraint))
    return y
