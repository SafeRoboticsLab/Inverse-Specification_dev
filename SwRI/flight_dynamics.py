#Template Processor

# uses $name to get and put variables -

import kajiki
import sys
import shortuuid
import os
import shutil
import numpy as np
import subprocess


class TemplateProcessor():

  def __init__(self, template_file):
    with open(template_file, "r") as myfile:
      data = myfile.read()
      #print(data)
      self.TextTemplate = kajiki.TextTemplate(data)
    myfile.close()
    self.data = data

  def writefile(self, indict, outfilename):
    self.outfilename = outfilename
    data = self.TextTemplate(indict).render()
    with open(self.outfilename, "w") as outfile:
      outfile.write(data)
      outfile.flush()


class MetricsReader():

  def __init__(self, metrics_file):
    self.metrics_file = metrics_file

  def read(self):
    self.outputs = {}
    with open(self.metrics_file, "r") as readfile:
      data = readfile.readlines()
      for line in data:
        if '#' not in line:
          cleanline = line.replace('(', '').replace(')', '')
          cleanline = cleanline.replace('/', '').replace('\n', '')
          # print(cleanline)
          splits = cleanline.split()
          # print(splits)
          # if len(splits) > 1:
          if len(splits) == 2:
            self.outputs[splits[0]] = splits[1:]
    return self.outputs


class SWRIFlightDynamics():

  def __init__(self, **kwargs):
    if 'template_file' not in kwargs:
      print(
          "DEXTER: ERROR (SWRI FlightDynamics)-",
          "Template file cannot be found "
      )
      sys.exit()
    else:
      self.template_file = kwargs['template_file']
    self.templateprocessor = TemplateProcessor(self.template_file)

    # copy the executable to the current directory
    if 'exec_file' not in kwargs:
      mydir = os.path.dirname(os.path.realpath(__file__))
      self.exec_file = os.path.join(mydir, 'new_fdm')
    else:
      self.exec_file = kwargs['exec_file']
    self.output_file = "metrics.out"
    self.cwd = os.getcwd()
    self.run_number = 0

  def sim(self, x, delete_folder=False, **kwargs):
    # implementation of evaluation method in DesignSpace class expects an
    # error if variables are arrays to iterate over values of array - so
    # raise type error if we get arrays in the input dictionary (fortran
    # code will fail otherwise)
    if any([
        isinstance(val, list) or isinstance(val, np.ndarray)
        for val in list(x.values())
    ]):
      raise TypeError

    self.run_number = self.run_number + 1
    print("Run# ", self.run_number)

    run_folder = os.path.join("tempStoreSim", "dex_" + shortuuid.uuid())
    os.makedirs(run_folder)
    directory = os.path.join(self.cwd, run_folder)

    # we copy the executable to the current runfolder for each run - this
    # is slow but guarantees thread safety
    shutil.copyfile(self.exec_file, os.path.join(directory, 'exec_file'))
    shutil.copystat(self.exec_file, os.path.join(directory, 'exec_file'))

    tmp_file_in = "input.inp"
    tmp_file_out = "output.out"
    self.templateprocessor.writefile(x, os.path.join(directory, tmp_file_in))
    bashCommand = "./exec_file" + "< " + tmp_file_in + " > " + tmp_file_out
    print("Started Sim")
    p = subprocess.run(bashCommand, cwd=directory, shell=True)
    if p.returncode != 0:
      print('ERROR during execution of fdm code!')
      sys.exit()
    print("Finished Sim")

    postprocess = MetricsReader(os.path.join(directory, self.output_file))
    y = postprocess.read()
    for key in y:
      y[key] = y[key][0]
      try:
        y[key] = float(y[key])
      except ValueError:
        pass

    # delete folder
    if delete_folder:
      shutil.rmtree(directory)
    return y
