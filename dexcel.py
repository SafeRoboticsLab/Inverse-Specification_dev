import os


os.sys.path.append(os.path.join(os.getcwd(), 'src'))

from auv.auv_sim import DexcelInterface

# initialize the interface object
filepath = os.path.join('src', 'AUV', 'AUVRangeSpeedV3.pkl')
ff_obj = DexcelInterface(filename=filepath)

# inputs
print("----- INPUTS -----")
print(ff_obj.inputs)
print("----- OUTPUTS -----")
print(ff_obj.outputs)

# To run set up a input dictionary - parameter names are in config.dexcel
x = {}

x['Battery_Fraction'] = 0.6

# lets set the battery fraction to 0.6 and run it various materials

for i in range(4):
  x['PV_Material_Choice'] = i + 1

  # evaluate the corpus -- keeping everything else as same as the excel sheet
  print("----------------------------------")
  print("INPUTS set:", x)
  print("----------------------------------")
  y = ff_obj.function(x)
  print("----------------------------------")
  print("OUTPUTS Computed")
  print("----------------------------------")
  print(y)
  # returns values of all the outputs

  print("----------------------------------")
  print("Objectives ")
  print(ff_obj.objectives)
  print(" Constraints")
  print(ff_obj.constraints)
  print("----------------------------------")

  out = ff_obj.problem(x)
  print(out)
