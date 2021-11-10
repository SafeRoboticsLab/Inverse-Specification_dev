#Template Processor

# uses $name to get and put variables -

import kajiki
import sys
import shortuuid
import os
import shutil
import numpy as np
import subprocess

from functools import partial
from multiprocessing.dummy import Pool


def remove_file(file_path):
  if os.path.isfile(file_path):
    os.remove(file_path)


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

  def __init__(self, template_file, exec_file, **kwargs):
    self.template_file = template_file
    self.templateprocessor = TemplateProcessor(self.template_file)
    self.exec_file = exec_file
    self.output_file = "metrics.out"
    self.cwd = os.getcwd()
    self.run_number = 0

  def sim(self, x, delete_folder=True, verbose=False, **kwargs):
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
    if verbose:
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
    if verbose:
      print("Started Sim")
    p = subprocess.run(bashCommand, cwd=directory, shell=True)
    if p.returncode != 0:
      print('ERROR during execution of fdm code!')
      sys.exit()
    if verbose:
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


class SWRIFlightDynamicsParallel():
  """
  This creates a simulator using multiple threads to speed up forward
  simulation.

  Reference: https://stackoverflow.com/questions/14533458/python-threading-multiple-bash-subprocesses
  """

  def __init__(
      self, template_file, exec_file, num_workers, prefix="dex_", **kwargs
  ):
    self.template_file = template_file
    self.templateprocessor = TemplateProcessor(self.template_file)
    self.exec_file = exec_file
    self.output_file = "metrics.out"
    self.tmp_file_in = "input.inp"
    self.tmp_file_out = "output.out"
    self.cwd = os.getcwd()
    self.run_number = 0
    self.num_workers = num_workers
    self.prefix = prefix

  def _create_work_directories(self, input_tuple):
    x, idx = input_tuple
    run_folder = os.path.join("tempStoreSim", self.prefix + str(idx))
    directory = os.path.join(self.cwd, run_folder)
    if not os.path.isdir(run_folder):
      os.makedirs(run_folder)
    shutil.copyfile(self.exec_file, os.path.join(directory, 'exec_file'))
    shutil.copystat(self.exec_file, os.path.join(directory, 'exec_file'))

    input_file = os.path.join(directory, self.tmp_file_in)
    self.templateprocessor.writefile(x, input_file)
    return directory

  def _sim(self, directory):
    bashCommand = (
        "./exec_file" + "< " + self.tmp_file_in + " > " + self.tmp_file_out
    )
    p = subprocess.run(bashCommand, cwd=directory, shell=True)
    if p.returncode != 0:
      print(directory)
      print('ERROR during execution of fdm code!')
      sys.exit()

    postprocess = MetricsReader(os.path.join(directory, self.output_file))
    y = postprocess.read()
    for key in y:
      y[key] = y[key][0]
      try:
        y[key] = float(y[key])
      except ValueError:
        pass

    return y

  def _clear(self, directory):
    shutil.rmtree(directory)
    # remove_file(os.path.join(directory, self.output_file))
    # remove_file(os.path.join(directory, self.tmp_file_in))
    # remove_file(os.path.join(directory, self.tmp_file_out))
    # remove_file(os.path.join(directory, "namelist.out"))
    # remove_file(os.path.join(directory, "path.out"))
    # remove_file(os.path.join(directory, "path2.out"))
    # remove_file(os.path.join(directory, "score.out"))

  def sim(self, X, delete_folder=True, **kwargs):
    pool = Pool(self.num_workers)
    input_tuples = [(x, i) for x, i in zip(X, np.arange(len(X)))]
    directories = []
    for directory in pool.imap(self._create_work_directories, input_tuples):
      directories.append(directory)
    pool.close()
    pool.join()

    pool = Pool(self.num_workers)
    Y = []
    for y in pool.imap(self._sim, directories):
      Y.append(y)
    self.run_number = self.run_number + len(X)
    pool.close()
    pool.join()

    if delete_folder:
      pool = Pool(self.num_workers)
      for _ in pool.imap(self._clear, directories):
        pass
      pool.close()
      pool.join()
    return Y
