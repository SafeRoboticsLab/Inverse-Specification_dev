# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch.nn as nn


class NeuralNetwork(nn.Module):
  """
  Constructs a fully-connected neural network with flexible depth, width and
  activation function choices. The output is a scalar in (0, 1), representating
  how feasible is this (state, action) pair.
  """

  def __init__(
      self, dimList, actType='Tanh', verbose=False, outAct=nn.Sigmoid()
  ):
    """
    Args:
        dimList (int List): the dimension of each layer.
        actType (str, optional): the type of activation function.
            Defaults to 'Tanh'. Currently supports 'Sin', 'Tanh' and 'ReLU'.
        verbose (bool, optional): print info or not. Defaults to False.
    """
    super(NeuralNetwork, self).__init__()

    # Construct module list: if use `Python List`, the modules are not added
    # to computation graph. Instead, we should use `nn.ModuleList()`.
    self.moduleList = nn.ModuleList()
    numLayer = len(dimList) - 1
    for idx in range(numLayer):
      i_dim = dimList[idx]
      o_dim = dimList[idx + 1]

      self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
      if idx == numLayer - 1:
        # self.moduleList.append(nn.Sigmoid())
        self.moduleList.append(outAct)
      else:
        if actType == 'Tanh':
          self.moduleList.append(nn.Tanh())
        elif actType == 'ReLU':
          self.moduleList.append(nn.ReLU())
        else:
          raise ValueError(
              'Activation type ({:s}) is not included!'.format(actType)
          )
        # self.moduleList.append(nn.Dropout(p=.5))
    if verbose:
      print(self.moduleList)

    # Initalizes the weight
    self._initialize_weights()

  def forward(self, x):
    for m in self.moduleList:
      x = m(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()
