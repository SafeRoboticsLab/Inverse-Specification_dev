# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
import imageio
import pickle


#== DATA PROCESSING ==
def normalize(F, F_min=None, F_max=None):
  if F_min is None:
    F_min = np.min(F, axis=0)
  if F_max is None:
    F_max = np.max(F, axis=0)
  F_spacing = F_max - F_min

  return (F-F_min) / F_spacing


def unnormalize(_F, F_min, F_max):
  return _F * (F_max-F_min) + F_min


#== RANDOM SEED ==
def setSeed(seed_val=0, useTorch=False):
  np.random.seed(seed_val)
  random.seed(seed_val)

  if useTorch:
    import torch
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#== PROBABILITY ==
def MapAndMean(weightArray, prob):
  """Gets the MAP and MEAN of the weights.

  Args:
      weightArray (float array):  array of weights
      prob (float array):         array consisting of weights' probability

  Returns:
      w_map (float array):  mode of weights
      w_mean (float array): weighted average of weights
  """
  idx = np.argmax(prob)
  w_map = weightArray[idx].reshape(-1)
  w_mean = np.sum(np.diag(prob) @ weightArray, axis=0).reshape(-1)

  return w_map, w_mean


def entropy(prob):
  """Calculates the entropy given the probabilities.

  Args:
      prob (float array): array of probabilities

  Returns:
      entropy (float): entropy of the distribution.
  """
  return -np.sum(prob * np.log(prob))


#== INIT / RESET ==
def initExp(numWeight=50, dim=3, useL1Norm=False, shrink=.95):
  """
  Samples weights from 2-norm or 1-norm ball. Generates designs given the
  weights.

  Args:
      numWeight (int, optional):  the number of weights. Defaults to 50.
      dim (int, optional):        the dimension of a weight. Defaults to 3.
      useL1Norm (bool, optional): project weights to 1-norm ball. Defaults to
          False.
      shrink (float, optional):   the ratio to shrink designs' features except
          the optimal design with respect to the optimal weight. Defaults to
          0.95.

  Returns:
      weightArray (float array):      array of weights.
      w_opt (float array):            optimal weight.
      idx_opt (int):                  index of the optimal weight.
      designFeature (float array):    array of designs' features
  """
  #= Weight Space
  weightArray, idx_opt = setWeightSpace(
      numWeight=numWeight, dim=dim, useL1Norm=useL1Norm
  )
  #= Design Space
  # numDesign = weightArray.shape[0]
  designFeature = setDesignSpace(weightArray, idx_opt, shrink=shrink)

  if useL1Norm:  # project to ||w||_1 = 1
    L1Norm = np.linalg.norm(weightArray, axis=1, ord=1).reshape(-1, 1)
    weightArray /= L1Norm
    #designFeature *= L1Norm
  w_opt = weightArray[idx_opt].copy()

  return weightArray, w_opt, idx_opt, designFeature


#== WEIGHT SPACE / OPT WEIGHT ==
def setWeightSpace(numWeight=50, dim=3, useL1Norm=False):
  """
  Samples weights from 2-norm or 1-norm ball.

  Args:
      numWeight (int, optional): the number of weights. Defaults to 50.
      dim (int, optional): the dimension of a weight. Defaults to 3.
      useL1Norm (bool, optional): project weights to 1-norm ball. Defaults to
          False.

  Returns:
      weightArray (float array): array of weights.
      w_opt (float array): optimal weight.
      idx_opt (int): index of the optimal weight.
  """
  # ref: https://www.sciencedirect.com/science/article/pii/S0047259X10001211

  weightArray = np.random.normal(size=(numWeight, dim))
  weightArray = np.abs(weightArray)
  weightArray /= np.linalg.norm(weightArray, axis=1, ord=2).reshape(-1, 1)
  # print(
  #     "The shape of the weight array is {:d} x {:d}.".format(
  #         weightArray.shape[0], weightArray.shape[1]
  #     )
  # )

  idx_opt = np.random.choice(numWeight)

  return weightArray, idx_opt


#== DESIGN SPACE ==
def setDesignSpace(weightArray, idx_opt, shrink=0.95):
  """
  Generates designs given the weights.

  Args:
      weightArray (float array): array of weights.
      idx_opt (int): index of the optimal weight.
      shrink (float, optional): the ratio to shrink designs' features except
          the optimal design with respect to the optimal weight. Defaults to
          0.95.

  Returns:
      designFeature (float array): array of designs' features
  """
  numWeight = weightArray.shape[0]
  designFeature = weightArray.copy()
  designFeature[np.arange(numWeight) != idx_opt, :] *= shrink

  return designFeature


def findInfeasibleDesigns(designFeature, activeConstraintSet):
  """
  Finds designs that don't meet the active constraints. The constraint is
  expressed as 1{feature < threshold}. If this bool expression is True, this
  design is infeasible.

  Args:
      designFeature (float array): designs' features.
      activeConstraintSet (int array): active constraints. 1st col.: feature
          index; 2nd col.: threshold.

  Returns:
      np.ndarray: bool, indicates which design is infeasible.
  """

  numDesign = designFeature.shape[0]
  infeasibleIndicator = np.full(shape=(numDesign,), fill_value=False)
  if activeConstraintSet is not None:
    for i in range(numDesign):
      flag = False
      for (featureIdx, threshold) in activeConstraintSet:
        upperBound = False
        if featureIdx[0] == '-':
          featureIdx = int(featureIdx[1:])
          upperBound = True
        featureIdx = int(featureIdx)
        if upperBound:
          if designFeature[i, featureIdx] > threshold:
            flag = True
            break
        else:
          if designFeature[i, featureIdx] < threshold:
            flag = True
            break
      infeasibleIndicator[i] = flag

  return infeasibleIndicator


#== PLOT ==
def plotProb(
    weightArray, prob, w_opt=None, w_proxy=None, title=None, numbering=False
):
  """
  Generates the scatter plot of weights with color specifies the probability.
  Specifies the optimal and/or proxy weight if provided. If the dimenson of
  weights is bigger than three, just use the first three dimension to show the
  distribution.

  Args:
      weightArray (float array): array of weights
      prob (float array): array consisting of weights' probability.
      w_opt (float array, optional): optimal weight. Defaults to None.
      w_proxy (float array, optional): proxy weight. Defaults to None.
      title (string, optional): title of the scatter plot. Defaults to None.
      numbering (bool, optional): show the index of each weight. Defaults to
          False.
  """
  plt.style.use('default')

  fig = plt.figure(figsize=(8, 4))
  ax = plt.axes(projection='3d')

  xs = weightArray[:, 0]
  ys = weightArray[:, 1]
  zs = weightArray[:, 2]
  cs = prob

  flag = False

  if np.all(w_opt is not None):
    flag = True
    ax.scatter3D(
        w_opt[0], w_opt[1], w_opt[2], marker='s', c='g', s=100, label='Opt'
    )

  if np.all(w_proxy is not None):
    flag = True
    ax.scatter3D(
        w_proxy[0], w_proxy[1], w_proxy[2], marker='^', c='k', s=100,
        label='Proxy'
    )

  # color by prob
  probMean = np.mean(prob)
  probStd = np.std(prob)
  if probStd == 0:
    probStd = 5e-3
  minTmp = probMean - 3.*probStd
  maxTmp = probMean + 3.*probStd
  sca = ax.scatter3D(
      xs, ys, zs, c=cs, cmap=cm.coolwarm, vmin=minTmp, vmax=maxTmp, alpha=1,
      edgecolors='k'
  )

  if numbering:
    for i, w in enumerate(weightArray):
      x, y, z = w
      ax.text(x, y, z, '{:d}'.format(i))

  if title is not None:
    fig.suptitle(title)

  ax.set_ylim(1, 0)
  #ax.invert_yaxis()

  ax.set_xlabel("$w_0$")
  ax.set_ylabel("$w_1$")
  ax.set_zlabel("$w_2$")

  if flag:
    ax.legend()

  fig.colorbar(sca, fraction=.2, pad=.1, shrink=.9)
  fig.show()


def plotEVSIMtx(EVSI_mtx):
  numDesign = EVSI_mtx.shape[0]

  fig, ax = plt.subplots(1, 1, figsize=(6, 6))
  im = ax.imshow(EVSI_mtx, cmap=cm.coolwarm)
  ax.figure.colorbar(im)

  ax.set_yticks(np.arange(numDesign))
  ax.set_xticks(np.arange(numDesign))
  ax.set_title("EVSI")
  ax.set_xlabel("Design Index")
  ax.set_ylabel("Design Index")

  plt.show()


def plotLoss(
    lossRecord, lw=0.2, fs=14, topClf=20, topReg=10, plotFigure=True,
    figPath=None
):

  fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

  for ax in axes:
    ax.set_xlabel("Iteration", fontsize=fs - 2)
    ax.set_ylabel("Amplitude", fontsize=fs - 2)
    ax.set_xlim(left=0, right=lossRecord.shape[0])

  ax = axes[0]
  ax.plot(lossRecord[:, 0], 'b-', linewidth=lw)
  ax.set_title("Total Loss", fontsize=fs)
  ax.set_ylim(bottom=0, top=topClf)

  ax = axes[1]
  ax.plot(lossRecord[:, 1], 'b-', linewidth=lw)
  ax.set_title("Classification Loss", fontsize=fs)
  ax.set_ylim(bottom=0, top=topClf)

  ax = axes[2]
  ax.plot(lossRecord[:, 2], 'b-', linewidth=lw)
  ax.set_title("Regularization", fontsize=fs)
  ax.set_ylim(bottom=0, top=topReg)

  plt.tight_layout()
  if plotFigure:
    plt.show()
    plt.pause(0.1)
  if figPath is not None:
    fig.savefig(figPath)
  plt.close()


#== PLOT CONFIDENCE INTERVAL ==
def cal_mean_std(array):
  arrayMean = np.mean(array, axis=1)
  arrayStd = np.std(array, axis=1)
  return arrayMean, arrayStd


def confidenceInterval(data, size, ratio):
  df = size - 1
  if df == 4:
    if ratio == 95:
      t = 2.776
    elif ratio == 98:
      t = 3.747
  elif df == 9:
    if ratio == 95:
      t = 2.26
    elif ratio == 98:
      t = 2.82
  elif df == 49:
    if ratio == 95:
      t = 2.01
    elif ratio == 98:
      t = 2.40
  elif df == 99:
    if ratio == 95:
      t = 1.984
    elif ratio == 98:
      t = 2.364
  dataMean = np.mean(data[:, :], axis=1)
  dataStd = np.std(data[:, :], axis=1)
  ERB = t * dataStd / np.sqrt(size)
  return dataMean, dataMean - ERB, dataMean + ERB


def plotMeanConfInterval(
    ax, x, stat, color, label, alpha, sz=6, detailLabel=False
):
  labelMean = label + ', Mean'
  labelCI = label + ', 95% CI'

  if detailLabel:
    ax.plot(x, stat[0], '-o', c=color, label=labelMean, ms=sz)
    ax.fill(
        np.concatenate([x, x[::-1]]), np.concatenate([stat[1], stat[2][::-1]]),
        alpha=alpha, fc=color, ec='None', label=labelCI
    )
  else:
    ax.plot(x, stat[0], '-o', c=color, label=label, ms=sz)
    ax.fill(
        np.concatenate([x, x[::-1]]), np.concatenate([stat[1], stat[2][::-1]]),
        alpha=alpha, fc=color, ec='None'
    )


#== PLOT GA POPULATION BY OBJECTIVES ==
def plotResultPairwise(
    n_obj, F, objective_names, axis_bound, nColDefault=5, subfigSz=4, fsz=16,
    sz=20, lw=3, activeConstraintSet=None
):

  def _getAx(idx, nRow, nCol, axArray):
    rowIdx = int(idx / nCol)
    colIdx = idx % nCol
    if nRow > 1:
      ax = axArray[rowIdx, colIdx]
    elif nCol > 1:
      ax = axArray[colIdx]
    else:
      ax = axArray
    return ax

  def _numPrevPlots(n_obj, obj_x_idx, obj_y_idx):
    return int((2*n_obj - obj_x_idx - 1) * obj_x_idx / 2
              ) + (obj_y_idx-obj_x_idx)

  numSnapshot = int(n_obj * (n_obj-1) / 2)
  if numSnapshot < 5:
    nCol = numSnapshot
  else:
    nCol = nColDefault
  nRow = int(np.ceil(numSnapshot / nCol))
  figsize = (nCol * subfigSz, nRow * subfigSz)
  fig, axArray = plt.subplots(nRow, nCol, figsize=figsize)
  if activeConstraintSet is not None:
    for (featureIdx, threshold) in activeConstraintSet:
      if featureIdx[0] == '-':
        featureIdx = int(featureIdx[1:])
      else:
        featureIdx = int(featureIdx)
      vmin, vmax = axis_bound[featureIdx]
      value = vmin + (vmax-vmin) * threshold
      for i in range(featureIdx):
        idx = _numPrevPlots(n_obj, i, featureIdx) - 1
        ax = _getAx(idx, nRow, nCol, axArray)
        ax.plot(axis_bound[i], [value, value], 'r--', lw=lw)
      for i in range(featureIdx + 1, n_obj):
        idx = _numPrevPlots(n_obj, featureIdx, i) - 1
        ax = _getAx(idx, nRow, nCol, axArray)
        ax.plot([value, value], axis_bound[i], 'r--', lw=lw)

  idx = 0
  for i in range(n_obj):
    for j in range(i + 1, n_obj):
      ax = _getAx(idx, nRow, nCol, axArray)
      ax.scatter(F[:, i], F[:, j], c='b', s=sz, alpha=0.5)
      ax.set_xlabel(objective_names['o' + str(i + 1)], fontsize=fsz)
      ax.set_ylabel(objective_names['o' + str(j + 1)], fontsize=fsz)
      ax.set_xlim(axis_bound[i, 0], axis_bound[i, 1])
      ax.set_ylim(axis_bound[j, 0], axis_bound[j, 1])
      idx += 1
  return fig


def plotResult3D(
    F, objective_names, axis_bound, fsz=16, sz=10, lw=2, azim=210, figsz=6
):
  indices = np.argsort(F[:, 0])
  F = F[indices]
  # bottom = np.mean(axis_bound[2,:])
  bottom = axis_bound[2, 0]

  fig = plt.figure(figsize=(figsz, figsz))
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  markerline, stemlines, baseline = ax.stem(
      F[:, 0], F[:, 1], F[:, 2], bottom=bottom, orientation='z', linefmt='k:',
      markerfmt='ko', basefmt='r-'
  )
  markerline.set_markerfacecolor('none')
  markerline.set_markersize(sz)
  stemlines.set_linewidth(lw)

  ax.set_xlim(axis_bound[0, 0], axis_bound[0, 1])
  ax.set_ylim(axis_bound[1, 0], axis_bound[1, 1])
  ax.set_zlim(axis_bound[2, 0], axis_bound[2, 1])
  ax.set_xlabel(objective_names['o1'], fontsize=fsz)
  ax.set_ylabel(objective_names['o2'], fontsize=fsz)
  ax.set_zlabel(objective_names['o3'], fontsize=fsz)
  ax.xaxis.set_major_locator(LinearLocator(3))
  ax.xaxis.set_major_formatter('{x:.0f}')
  ax.yaxis.set_major_locator(LinearLocator(3))
  ax.yaxis.set_major_formatter('{x:.1f}')
  ax.zaxis.set_major_locator(LinearLocator(3))
  ax.zaxis.set_major_formatter('{x:.0f}')
  ax.azim = azim
  return fig


def generateGIF(
    dataFolder, destFolder, fn='', checkGeneration=1, maxGen=200, second=20
):

  # collect figures
  figProgress = os.path.join(dataFolder, 'figure', 'progress')
  fileList = []
  numFigure = int(maxGen / checkGeneration)
  for i in range(numFigure):
    idx = 1 + checkGeneration*i
    fileList.append(os.path.join(figProgress, str(idx) + '.png'))
  print(len(fileList))

  # generate GIF
  gifPath = os.path.join(destFolder, fn + 'progress.gif')
  print(gifPath)
  fps = int(numFigure / second)
  images = []
  for filename in fileList:
    images.append(imageio.imread(filename))

  imageio.mimsave(gifPath, images, loop=1, fps=fps)


#== PLOT INFERENCE OUTPUT ==
def getInferenceOutput(agent, nx, ny, obj_list):
  nx = 101
  ny = 101

  xs = np.linspace(0, 1, nx)
  ys = np.linspace(0, 1, ny)
  nz = len(obj_list)

  X, Y = np.meshgrid(xs, ys, indexing='ij')
  Z = np.empty(shape=(nx, ny, nz), dtype=float)

  for i in range(ny):
    print(i, end='\r')
    F = np.empty(shape=(nx, 3), dtype=float)
    F[:, 0] = X[:, i]
    F[:, 1] = Y[:, i]
    for j, tmp in enumerate(obj_list):
      F[:, 2] = tmp
      rwd = agent.inference.eval(F)
      Z[:, i, j] = rwd
  return X, Y, Z


def plotOutput3D(
    X, Y, Z, obj_list_un, axis_bound, fsz=16, subfigSz=4, cm='coolwarm',
    alpha=1
):

  nRow = 1
  nCol = len(obj_list_un)
  figsize = (nCol * subfigSz, nRow * subfigSz)

  nticks = 3
  xticklabels = np.linspace(axis_bound[0, 0], axis_bound[0, 1], nticks)
  xticklabels = [int(x) for x in xticklabels]
  yticklabels = np.linspace(axis_bound[1, 0], axis_bound[1, 1], nticks)
  ticks = np.linspace(0, 1, nticks)

  vmin = np.floor(np.min(Z) / 0.1) * 0.1
  vmax = np.ceil(np.max(Z) / 0.1) * 0.1
  zticks = np.linspace(vmin, vmax, nticks)
  zticklabels = ['{:.1f}'.format(x) for x in zticks]

  fig = plt.figure(figsize=figsize)
  fig.set_facecolor('white')

  for idx, obj in enumerate(obj_list_un):
    ax = fig.add_subplot(nRow, nCol, idx + 1, projection='3d')
    ax.plot_surface(
        X, Y, Z[:, :, idx], cmap=cm, vmin=vmin, vmax=vmax, alpha=alpha,
        linewidth=0, antialiased=False
    )

    # Customize the z axis.
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels, fontsize=fsz - 4)

    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(ticks)
    ax.set_yticklabels(yticklabels, fontsize=fsz - 4)

    ax.set_zlim(vmin, vmax)
    ax.set_zticks(zticks)
    ax.set_zticklabels(zticklabels, fontsize=fsz - 4)

    if idx == nCol - 1:
      ax.set_xlabel('Range', fontsize=fsz)
      ax.set_ylabel('Speed', fontsize=fsz)
    ax.set_title('Power = {:.0f}'.format(-obj), fontsize=fsz, pad=0)
    ax.azim = 240
    ax.elev = 15
  return fig


def plotOutput2D(
    X, Y, Z, obj_list_un, axis_bound, levelRatios, fsz=22, subfigSz=4,
    cm='coolwarm', alpha=1, lw=2.5
):

  F_min = axis_bound[:, 0]
  F_max = axis_bound[:, 1]

  nRow = 1
  nCol = len(obj_list_un)
  figsize = (nCol * subfigSz, nRow * subfigSz)

  ticks = np.array([0., 0.2, 0.5, 1.])
  xticklabels = unnormalize(ticks, F_min[0], F_max[0])
  xticklabels = [int(x) for x in xticklabels]
  yticklabels = unnormalize(ticks, F_min[1], F_max[1])
  yticklabels = ['{:.2f}'.format(y) for y in yticklabels]
  extent = [0, 1, 0, 1]

  vmin = np.floor(np.min(Z) / 0.1) * 0.1
  vmax = np.ceil(np.max(Z) / 0.1) * 0.1
  levels = levelRatios * (vmax-vmin) + vmin

  fig, axes = plt.subplots(
      nRow, nCol, figsize=figsize, sharex=True, sharey=True
  )
  for idx, obj in enumerate(obj_list_un):
    ax = axes[idx]
    v = Z[:, :, idx]
    ax.imshow(
        v.T, cmap=cm, vmin=vmin, vmax=vmax, alpha=alpha, origin='lower',
        extent=extent
    )
    CS = ax.contour(
        X, Y, v, levels=levels, colors='k', linewidths=lw, linestyles='dashed'
    )
    ax.clabel(CS, fmt='%.2f', colors='k', fontsize=fsz - 6)

    # Customize the z axis.
    ax.set_xlim(0., 1.)
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels, fontsize=fsz - 4)

    ax.set_ylim(0., 1.)
    ax.set_yticks(ticks)
    ax.set_yticklabels(yticklabels, fontsize=fsz - 4)

    ax.set_title('Power = {:.0f}'.format(-obj), fontsize=fsz, pad=0)

  fig.supxlabel('Range', fontsize=fsz)
  axes[0].set_ylabel('Speed', fontsize=fsz)
  return fig, axes


#== pickle ==
def save_obj(obj, filename):
  with open(filename + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  with open(filename + '.pkl', 'rb') as f:
    return pickle.load(f)
