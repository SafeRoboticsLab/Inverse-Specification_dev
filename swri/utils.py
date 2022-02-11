import os
import matplotlib.pyplot as plt

from utils import plot_result_pairwise, plot_single_objective


def plot_pop_swri(
    features, scores, component_values, objective_names, input_names_dict,
    objectives_bound=None, scores_bound=None, component_values_bound=None,
    c='b', alpha=0.5, show_legend=True, feature_centers=None, cmap='tab20b'
):
  fig1 = plot_result_pairwise(
      len(objective_names), features, objective_names,
      axis_bound=objectives_bound, c=c, alpha=alpha, show_legend=show_legend,
      centers=feature_centers, cmap=cmap
  )

  fig2 = plot_single_objective(
      scores, dict(o1="True Scores"), axis_bound=scores_bound, c=c,
      alpha=alpha, show_legend=show_legend, cmap=cmap
  )

  fig3 = plot_result_pairwise(
      len(input_names_dict), component_values, input_names_dict,
      axis_bound=component_values_bound, n_col_default=5, subfigsz=4, fsz=16,
      sz=20, c=c, alpha=alpha, show_legend=show_legend, cmap=cmap
  )

  return fig1, fig2, fig3


def report_pop_swri(
    obj, fig_progress_folder, n_acc_fb, objective_names, input_names_dict,
    objectives_bound=None, scores_bound=None, component_values_bound=None,
    c='b', alpha=0.5
):
  n_gen = obj.n_gen
  n_nds = len(obj.opt)
  CV = obj.opt.get('CV').min()
  print(f"gen[{n_gen}]: n_nds: {n_nds} CV: {CV}")

  features = -obj.opt.get('F')
  component_values = obj.opt.get('X')
  scores = obj.opt.get('scores').reshape(-1)
  print(scores)

  fig1, fig2, fig3 = plot_pop_swri(
      features, scores, component_values, objective_names, input_names_dict,
      objectives_bound, scores_bound, component_values_bound, c=c, alpha=alpha
  )

  fig1.supxlabel(
      'G{}: {} cumulative queries'.format(n_gen, n_acc_fb), fontsize=20
  )
  fig1.tight_layout()
  fig1.savefig(os.path.join(fig_progress_folder, str(n_gen) + '.png'))

  fig2.supxlabel(str(n_gen), fontsize=20)
  fig2.tight_layout()
  fig2.savefig(
      os.path.join(fig_progress_folder, 'score_' + str(n_gen) + '.png')
  )

  fig3.supxlabel(str(n_gen), fontsize=20)
  fig3.tight_layout()
  fig3.savefig(
      os.path.join(fig_progress_folder, 'inputs_' + str(n_gen) + '.png')
  )

  plt.close('all')
  return features, component_values, scores
