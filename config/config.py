import yaml

# usage:
# configDict = load_config(filePath)
# dump_config(filePath, objects, keys)


class Struct:

  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


def dict2object(dictionary, key):
  return Struct(**dictionary[key])


def load_config(filePath):
  with open(filePath) as f:
    data = yaml.safe_load(f)
  configDict = {}
  for key, value in data.items():
    configDict[key] = Struct(**value)
  return configDict


def dump_config(filePath, objects, keys):
  data = {}
  for key, object in zip(keys, objects):
    data[key] = object.__dict__
  with open(filePath, "w") as f:
    yaml.dump(data, f)
