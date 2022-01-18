import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from swri.problem import SWRIProblem
from swri.utils import plot_pop_swri
from utils import plot_single_objective
from clustering import get_cluster


def main(args, idx_range):
  with open(args.data, 'rb') as input:
    designs_collection = pickle.load(input)
  out_folder = os.path.join('dataset', args.out_folder)
  os.makedirs(out_folder, exist_ok=True)

  component_values = designs_collection['component_values'][
      idx_range[0]:idx_range[1]]
  features = designs_collection['features'][idx_range[0]:idx_range[1]]
  scores = designs_collection['scores'][idx_range[0]:idx_range[1]]
  predicted_scores = designs_collection['predicted_scores'][
      idx_range[0]:idx_range[1]]

  # region: problem metrics
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblem(
      TEMPLATE_FILE, EXEC_FILE, num_workers=5,
      prefix="eval_" + time.strftime("%m-%d-%H_%M") + "_"
  )

  objective_names = problem.objective_names
  obj_indicator = problem.obj_indicator
  scores_bound = np.array([-1e-8, 430])
  input_names_dict = {}
  for i in range(len(problem.input_names)):
    input_names_dict['o' + str(i + 1)] = problem.input_names[i][8:]
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  # endregion

  # clustering
  kmax = 10
  kmeans, sil_vec_max, sil_max_idx, sil = get_cluster(
      features, kmax=kmax, get_sil_curve=True
  )
  labels = kmeans.labels_
  print(kmeans.cluster_centers_)
  dist2cluster_centers = np.empty(shape=(features.shape[0], sil_max_idx))
  for i, center in enumerate(kmeans.cluster_centers_):
    dist2cluster_centers[:, i] = np.linalg.norm(features - center, axis=1)
  designs_collection["dist2cluster_centers"] = dist2cluster_centers
  pickle_path = os.path.join(out_folder, 'design_collections.pkl')
  with open(pickle_path, 'wb') as output:
    pickle.dump(designs_collection, output, pickle.HIGHEST_PROTOCOL)

  # region: plotting
  subfigsz = 4
  fsz = 16
  sz = 20
  cmap = 'Accent'
  alpha = 1.

  fig6, ax = plt.subplots(1, 1, figsize=(subfigsz, subfigsz))
  ax.plot(np.arange(2, kmax + 1), sil, 'b-o', lw=2)
  ax.set_ylabel("Silhouette scores", fontsize=fsz)
  ax.set_xlabel("The number of clusters", fontsize=fsz)
  fig6.tight_layout()
  fig6.savefig(os.path.join(out_folder, 'silhouette.png'))

  features_plotted = (-features) * obj_indicator
  objectives_bound_plotted = np.array([
      [-160, 4000],
      [-16, 400],
      [-1.2, 30],
      [-0.12, 3],
      [-0.06, 1.5],
  ])
  feature_centers = (-kmeans.cluster_centers_) * obj_indicator

  if args.no_cluster:
    fig1, fig2, fig3 = plot_pop_swri(
        features_plotted, scores, component_values, objective_names,
        input_names_dict, objectives_bound_plotted, scores_bound,
        component_values_bound, c='b', alpha=alpha, show_legend=False
    )
  else:
    fig1, fig2, fig3 = plot_pop_swri(
        features_plotted, scores, component_values, objective_names,
        input_names_dict, objectives_bound_plotted, scores_bound,
        component_values_bound, c=labels, alpha=alpha, show_legend=True,
        feature_centers=feature_centers, cmap=cmap
    )

  fig1.tight_layout()
  fig1.savefig(os.path.join(out_folder, 'features.png'))

  fig2.tight_layout()
  fig2.savefig(os.path.join(out_folder, 'scores.png'))

  fig3.tight_layout()
  fig3.savefig(os.path.join(out_folder, 'component_values.png'))

  if args.no_cluster:
    fig4 = plot_single_objective(
        predicted_scores, dict(o1="Predicted Scores"),
        axis_bound=[0, 1 + 1e-8], c='b', alpha=alpha
    )
  else:
    fig4 = plot_single_objective(
        predicted_scores, dict(o1="Predicted Scores"),
        axis_bound=[0, 1 + 1e-8], c=labels, alpha=alpha, cmap=cmap
    )
  fig4.tight_layout()
  fig4.savefig(os.path.join(out_folder, 'scores_predicted.png'))

  fig5, ax = plt.subplots(1, 1, figsize=(subfigsz, subfigsz))
  if args.no_cluster:
    scatter = ax.scatter(
        scores.reshape(-1), predicted_scores.reshape(-1), c='b', alpha=alpha,
        s=sz
    )
  else:
    scatter = ax.scatter(
        scores.reshape(-1), predicted_scores.reshape(-1), c=labels,
        alpha=alpha, s=sz, cmap=cmap
    )
    legend1 = ax.legend(
        *scatter.legend_elements(), loc="best", title="Classes"
    )
    ax.add_artist(legend1)
  line_x = np.array([scores_bound[0], scores_bound[1]])
  line_y = np.array([0., 1 + 1e-8])
  ax.plot(line_x, line_y, 'r-.', linewidth=2)
  ax.set_ylabel("Predicted Scores", fontsize=fsz)
  ax.set_xlabel("True Scores", fontsize=fsz)
  ax.set_xlim(scores_bound[0], scores_bound[1])
  ax.set_ylim(0., 1. + 1e-8)
  fig5.tight_layout()
  fig5.savefig(os.path.join(out_folder, 'scores_comparison.png'))
  plt.close('all')
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", help="data path", type=str)
  parser.add_argument("-o", "--out_folder", help="output folder", type=str)
  parser.add_argument(
      "-n", "--no_cluster", help="w/o clustering", action="store_true"
  )
  parser.add_argument(
      "-id", "--idx_range", help="index range", default=[0, 50], nargs=2,
      type=int
  )
  args = parser.parse_args()
  main(args, idx_range=args.idx_range)
