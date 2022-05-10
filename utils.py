# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

from typing import List, Tuple, Any, Optional
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
import imageio
import pickle
from pymoo.core.problem import Problem

from humansim.human_simulator import HumanSimulator
from invspec.design import Design
from invspec.inv_spec import InvSpec


# region: DATA PROCESSING
def normalize(input, input_min=None, input_max=None):
  if input_min is None:
    input_min = np.min(input, axis=0)
  if input_max is None:
    input_max = np.max(input, axis=0)
  F_spacing = input_max - input_min

  return (input-input_min) / F_spacing


def unnormalize(_F, input_min, input_max):
  return _F * (input_max-input_min) + input_min


# endregion


# region: RANDOM SEED
def set_seed(seed_val=0, use_torch=False):
  np.random.seed(seed_val)
  random.seed(seed_val)

  if use_torch:
    import torch
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# endregion


# region: PROBABILITY
def get_map_mean(weight_array, prob):
  """Gets the MAP and MEAN of the weights.

  Args:
      weight_array (float array):  array of weights
      prob (float array):         array consisting of weights' probability

  Returns:
      w_map (float array):  mode of weights
      w_mean (float array): weighted average of weights
  """
  idx = np.argmax(prob)
  w_map = weight_array[idx].reshape(-1)
  w_mean = np.sum(np.diag(prob) @ weight_array, axis=0).reshape(-1)

  return w_map, w_mean


def entropy(prob):
  """Calculates the entropy given the probabilities.

  Args:
      prob (float array): array of probabilities

  Returns:
      entropy (float): entropy of the distribution.
  """
  return -np.sum(prob * np.log(prob))


# endregion


# region: CONFIDENCE INTERVAL
def cal_mean_std(array):
  array_mean = np.mean(array, axis=1)
  array_std = np.std(array, axis=1)
  return array_mean, array_std


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
  data_mean = np.mean(data[:, :], axis=1)
  data_std = np.std(data[:, :], axis=1)
  ERB = t * data_std / np.sqrt(size)
  return data_mean, data_mean - ERB, data_mean + ERB


# endregion


# region: PLOT
def plot_prob(
    weight_array, prob, w_opt=None, w_proxy=None, title=None, numbering=False
):
  """
  Generates the scatter plot of weights with color specifies the probability.
  Specifies the optimal and/or proxy weight if provided. If the dimenson of
  weights is bigger than three, just use the first three dimension to show the
  distribution.

  Args:
      weight_array (float array): array of weights
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

  xs = weight_array[:, 0]
  ys = weight_array[:, 1]
  zs = weight_array[:, 2]
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
  prob_mean = np.mean(prob)
  prob_std = np.std(prob)
  if prob_std == 0:
    prob_std = 5e-3
  min_tmp = prob_mean - 3.*prob_std
  max_tmp = prob_mean + 3.*prob_std
  sca = ax.scatter3D(
      xs, ys, zs, c=cs, cmap=cm.coolwarm, vmin=min_tmp, vmax=max_tmp, alpha=1,
      edgecolors='k'
  )

  if numbering:
    for i, w in enumerate(weight_array):
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


def plot_EVSI_mtx(EVSI_mtx):
  num_design = EVSI_mtx.shape[0]

  fig, ax = plt.subplots(1, 1, figsize=(6, 6))
  im = ax.imshow(EVSI_mtx, cmap=cm.coolwarm)
  ax.figure.colorbar(im)

  ax.set_yticks(np.arange(num_design))
  ax.set_xticks(np.arange(num_design))
  ax.set_title("EVSI")
  ax.set_xlabel("Design Index")
  ax.set_ylabel("Design Index")

  plt.show()


def plot_loss(
    loss_record, lw=0.2, fs=14, clf_loss_max=20, reg_loss_max=10,
    plot_figure=True, fig_path=None
):

  fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

  for ax in axes:
    ax.set_xlabel("Iteration", fontsize=fs - 2)
    ax.set_ylabel("Amplitude", fontsize=fs - 2)
    ax.set_xlim(left=0, right=loss_record.shape[0])

  ax = axes[0]
  ax.plot(loss_record[:, 0], 'b-', linewidth=lw)
  ax.set_title("Total Loss", fontsize=fs)
  ax.set_ylim(bottom=0, top=clf_loss_max)

  ax = axes[1]
  ax.plot(loss_record[:, 1], 'b-', linewidth=lw)
  ax.set_title("Classification Loss", fontsize=fs)
  ax.set_ylim(bottom=0, top=clf_loss_max)

  ax = axes[2]
  ax.plot(loss_record[:, 2], 'b-', linewidth=lw)
  ax.set_title("Regularization", fontsize=fs)
  ax.set_ylim(bottom=0, top=reg_loss_max)

  plt.tight_layout()
  if plot_figure:
    plt.show()
    plt.pause(0.1)
  if fig_path is not None:
    fig.savefig(fig_path)
  plt.close()


def plot_mean_conf_interval(
    ax, x, stat, color, label, alpha, sz=6, detail_label=False, lw=1.5
):
  label_mean = label + ', Mean'
  label_CI = label + ', 95% CI'

  if detail_label:
    ax.plot(x, stat[0], '-o', c=color, label=label_mean, ms=sz, lw=lw)
    ax.fill(
        np.concatenate([x, x[::-1]]), np.concatenate([stat[1], stat[2][::-1]]),
        alpha=alpha, fc=color, ec='None', label=label_CI
    )
  else:
    ax.plot(x, stat[0], '-o', c=color, label=label, ms=sz, lw=lw)
    ax.fill(
        np.concatenate([x, x[::-1]]), np.concatenate([stat[1], stat[2][::-1]]),
        alpha=alpha, fc=color, ec='None'
    )


# PLOT GA POPULATION BY OBJECTIVES
def plot_single_objective(
    F, objective_names, subfigsz=4, fsz=16, sz=20, axis_bound=None, c='b',
    alpha=0.5, show_legend=False, cmap='tab20b'
):
  F = F.reshape(-1)
  fig, ax = plt.subplots(1, 1, figsize=(subfigsz, subfigsz))
  # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
  # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
  scatter = ax.scatter(
      np.arange(F.shape[0]), F, c=c, s=sz, alpha=alpha, cmap=cmap
  )
  ax.set_ylabel(objective_names['o1'], fontsize=fsz)
  ax.set_xlabel("Design Index", fontsize=fsz)
  if axis_bound is not None:
    ax.set_ylim(axis_bound[0], axis_bound[1])
  if show_legend:
    legend1 = ax.legend(
        *scatter.legend_elements(), loc="best", title="Classes"
    )
    ax.add_artist(legend1)
  return fig


def plot_result_pairwise(
    n_obj, F, objective_names, axis_bound, n_col_default=5, subfigsz=4, fsz=16,
    sz=20, lw=3, active_constraint_set=None, c='b', alpha=0.5,
    show_legend=False, centers=None, cmap='tab20b'
):

  def _get_ax(idx, n_row, n_col, ax_array):
    rowIdx = int(idx / n_col)
    colIdx = idx % n_col
    if n_row > 1:
      ax = ax_array[rowIdx, colIdx]
    elif n_col > 1:
      ax = ax_array[colIdx]
    else:
      ax = ax_array
    return ax

  def _get_num_prev_plots(n_obj, obj_x_idx, obj_y_idx):
    return int((2*n_obj - obj_x_idx - 1) * obj_x_idx / 2
              ) + (obj_y_idx-obj_x_idx)

  num_snapshot = int(n_obj * (n_obj-1) / 2)
  if num_snapshot < 5:
    n_col = num_snapshot
  else:
    n_col = n_col_default
  n_row = int(np.ceil(num_snapshot / n_col))
  figsize = (n_col * subfigsz, n_row * subfigsz)
  fig, ax_array = plt.subplots(n_row, n_col, figsize=figsize)
  # for axi in ax_array.flat:
  #   axi.xaxis.set_major_locator(plt.MaxNLocator(5))
  #   axi.yaxis.set_major_locator(plt.MaxNLocator(5))
  if active_constraint_set is not None:
    assert axis_bound is not None, "constraint plotting requires axis bound"
    for (feature_idx, threshold) in active_constraint_set:
      if feature_idx[0] == '-':
        feature_idx = int(feature_idx[1:])
      else:
        feature_idx = int(feature_idx)
      vmin, vmax = axis_bound[feature_idx]
      value = vmin + (vmax-vmin) * threshold
      for i in range(feature_idx):
        idx = _get_num_prev_plots(n_obj, i, feature_idx) - 1
        ax = _get_ax(idx, n_row, n_col, ax_array)
        ax.plot(axis_bound[i], [value, value], 'r--', lw=lw)
      for i in range(feature_idx + 1, n_obj):
        idx = _get_num_prev_plots(n_obj, feature_idx, i) - 1
        ax = _get_ax(idx, n_row, n_col, ax_array)
        ax.plot([value, value], axis_bound[i], 'r--', lw=lw)

  idx = 0
  for i in range(n_obj):
    for j in range(i + 1, n_obj):
      ax = _get_ax(idx, n_row, n_col, ax_array)
      scatter = ax.scatter(F[:, i], F[:, j], c=c, s=sz, alpha=alpha, cmap=cmap)
      if centers is not None:
        ax.scatter(
            centers[:, i], centers[:, j], c=np.arange(len(centers)), s=64,
            alpha=1, marker='^', cmap=cmap
        )
      ax.set_xlabel(objective_names['o' + str(i + 1)], fontsize=fsz)
      ax.set_ylabel(objective_names['o' + str(j + 1)], fontsize=fsz)
      if axis_bound is not None:
        ax.set_xlim(axis_bound[i, 0], axis_bound[i, 1])
        ax.set_ylim(axis_bound[j, 0], axis_bound[j, 1])
      idx += 1
  if show_legend:
    ax = ax_array[n_row - 1][n_col - 1]
    legend1 = ax.legend(
        *scatter.legend_elements(), loc="best", title="Classes"
    )
    ax.add_artist(legend1)
  return fig


def plot_result_3D(
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


def generate_GIF(
    data_folder, dest_folder, fn='', check_generation=1, max_generation=200,
    second=20
):

  # collect figures
  figProgress = os.path.join(data_folder, 'figure', 'progress')
  fileList = []
  numFigure = int(max_generation / check_generation)
  for i in range(numFigure):
    idx = 1 + check_generation*i
    fileList.append(os.path.join(figProgress, str(idx) + '.png'))
  print(len(fileList))

  # generate GIF
  gif_path = os.path.join(dest_folder, fn + 'progress.gif')
  print(gif_path)
  fps = int(numFigure / second)
  images = []
  for filename in fileList:
    images.append(imageio.imread(filename))

  imageio.mimsave(gif_path, images, loop=1, fps=fps)


# PLOT INFERENCE OUTPUT
def get_inference_output(agent, nx, ny, obj_list):
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


def plot_output_3D(
    X, Y, Z, obj_list_un, axis_bound, fsz=16, subfigsz=4, cm='coolwarm',
    alpha=1
):

  n_row = 1
  n_col = len(obj_list_un)
  figsize = (n_col * subfigsz, n_row * subfigsz)

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
    ax = fig.add_subplot(n_row, n_col, idx + 1, projection='3d')
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

    if idx == n_col - 1:
      ax.set_xlabel('Range', fontsize=fsz)
      ax.set_ylabel('Speed', fontsize=fsz)
    ax.set_title('Power = {:.0f}'.format(-obj), fontsize=fsz, pad=0)
    ax.azim = 240
    ax.elev = 15
  return fig


def plot_output_2D(
    X, Y, Z, obj_list_un, axis_bound, level_ratios, fsz=22, subfigsz=4,
    cm='coolwarm', alpha=.8, lw=2.5
):
  cm = plt.cm.get_cmap(cm).reversed()

  input_min = axis_bound[:, 0]
  input_max = axis_bound[:, 1]

  n_row = 1
  n_col = len(obj_list_un)
  figsize = (n_col * subfigsz, n_row * subfigsz)

  ticks = np.array([0., 0.2, 0.5, 1.])
  xticklabels = unnormalize(ticks, input_min[0], input_max[0])
  xticklabels = [int(x) for x in xticklabels]
  yticklabels = unnormalize(ticks, input_min[1], input_max[1])
  yticklabels = ['{:.2f}'.format(y) for y in yticklabels]
  extent = [0, 1, 0, 1]

  vmin = np.floor(np.min(Z) / 0.1) * 0.1
  vmax = np.ceil(np.max(Z) / 0.1) * 0.1
  levels = level_ratios * (vmax-vmin) + vmin

  fig, axes = plt.subplots(
      n_row, n_col, figsize=figsize, sharex=True, sharey=True
  )
  for idx, obj in enumerate(obj_list_un):
    ax = axes[idx]
    v = Z[:, :, idx]
    im = ax.imshow(
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

    ax.set_title('Power = {:.0f} (W)'.format(-obj), fontsize=fsz, pad=0)

  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, vmax]
  )
  cbar.ax.tick_params(labelsize=fsz - 4)

  fig.supxlabel('Range (m)', fontsize=fsz)
  axes[0].set_ylabel('Speed (m/s)', fontsize=fsz)
  return fig, axes


# endregion


# region: pickle
def save_obj(obj, filename):
  with open(filename + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  with open(filename + '.pkl', 'rb') as f:
    return pickle.load(f)


# endregion


# region: random sampling and pymoo evaluate
def sample_and_evaluate(
    problem: Problem, component_values_bound: np.ndarray, num_samples: int = 8
):
  components = unnormalize(
      np.random.rand(num_samples, problem.n_var), component_values_bound[:, 0],
      component_values_bound[:, 1]
  )

  # get features
  y = {}
  problem._evaluate(components, y)
  return components, y


# endregion


# region: interaction with human
def get_infeasible_designs(
    fwd_features: np.ndarray,
    active_constraint_set: Optional[List[Tuple[str, float]]]
) -> np.ndarray:
  """
  Finds designs that don't meet the active constraints. The constraint is
  expressed as 1{feature < threshold}. If this bool expression is True, this
  design is infeasible.

  Args:
      fwd_features (float array): designs' features.
      active_constraint_set (list): active constraints. Each entry is a tuple
          of (feature_idx, threshold). Note that if the feature index starts
          with "-", this means this constraint is an upper bound.

  Returns:
      np.ndarray: bool, indicates which design is infeasible.
  """

  num_design = fwd_features.shape[0]
  infeasible_indicator = np.full(shape=(num_design,), fill_value=False)
  if active_constraint_set is not None:
    for i in range(num_design):
      flag = False
      for (feature_idx, threshold) in active_constraint_set:
        thr_as_upper_bound = False
        if feature_idx[0] == '-':
          feature_idx = int(feature_idx[1:])
          thr_as_upper_bound = True
        feature_idx = int(feature_idx)
        if thr_as_upper_bound:
          if fwd_features[i, feature_idx] > threshold:
            flag = True
            break
        else:
          if fwd_features[i, feature_idx] < threshold:
            flag = True
            break
      infeasible_indicator[i] = flag

  return infeasible_indicator


def query_and_collect(
    query: List[Design], query_key: int, human: HumanSimulator, agent: InvSpec,
    config_inv_spec: Any, collect_undistinguished: Optional[bool] = False
) -> Tuple[Optional[int], Optional[np.ndarray]]:
  # get feedback
  fb_raw = human.get_ranking(query, key=query_key)

  # store feedback
  inputs_to_invspec = np.array([
      design.get_test_metrics(query_key) for design in query
  ], dtype=np.float32)

  if config_inv_spec.INPUT_NORMALIZE:
    inputs_to_invspec = agent.inference.normalize(inputs_to_invspec)
  q_1 = (inputs_to_invspec[0:1, :], np.empty(shape=(1, 0)))
  q_2 = (inputs_to_invspec[1:2, :], np.empty(shape=(1, 0)))

  if fb_raw != 2:
    if fb_raw == 0:
      fb_invspec = 1
    elif fb_raw == 1:
      fb_invspec = -1
    agent.store_feedback(q_1, q_2, fb_invspec)
    return fb_invspec, inputs_to_invspec
  elif collect_undistinguished:
    eps = np.random.uniform()
    fb_invspec = 1 if eps > 0.5 else -1
    agent.store_feedback(q_1, q_2, fb_invspec)
    return fb_invspec, inputs_to_invspec
  else:
    return None, None


# endregion
