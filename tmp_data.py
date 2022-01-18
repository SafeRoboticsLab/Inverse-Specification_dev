import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from swri.problem import SWRIProblem
from swri.utils import plot_pop_swri
from utils import plot_single_objective


def main(args):
  with open(args.data, 'rb') as input:
    designs_collection = pickle.load(input)
  out_folder = os.path.join('dataset', args.out_folder)
  os.makedirs(out_folder, exist_ok=True)

  component_values = designs_collection['component_values']
  features = designs_collection['features']
  scores = designs_collection['scores']
  predicted_scores = designs_collection['predicted_scores']
  dist2cluster_centers = designs_collection["dist2cluster_centers"]

  # region: problem metrics
  TEMPLATE_FILE = os.path.join('swri', 'template', 'FlightDyn_quadH.inp')
  EXEC_FILE = os.path.join('swri', "new_fdm")
  problem = SWRIProblem(
      TEMPLATE_FILE, EXEC_FILE, num_workers=5,
      prefix="eval_" + time.strftime("%m-%d-%H_%M") + "_"
  )

  objective_names = problem.objective_names
  objectives_bound = np.array([
      [0, 4000],
      [-400, 0],
      [0, 30],
      [-50, 0.],
      [-12, 0.],
  ])
  scores_bound = np.array([-1e-8, 430])
  input_names_dict = {}
  for i in range(len(problem.input_names)):
    input_names_dict['o' + str(i + 1)] = problem.input_names[i][8:]
  component_values_bound = np.concatenate(
      (problem.xl[:, np.newaxis], problem.xu[:, np.newaxis]), axis=1
  )
  # endregion

  # clustering
  kmeans = KMeans(n_clusters=dist2cluster_centers.shape[1]).fit(features)
  labels = kmeans.labels_

  # region: plotting
  subfigsz = 4
  fsz = 16
  sz = 20

  fig1, fig2, fig3 = plot_pop_swri(
      features, scores, component_values, objective_names, input_names_dict,
      objectives_bound, scores_bound, component_values_bound, c=labels,
      alpha=0.8
  )

  fig1.tight_layout()
  fig1.savefig(os.path.join(out_folder, 'features.png'))

  fig2.tight_layout()
  fig2.savefig(os.path.join(out_folder, 'scores.png'))

  fig3.tight_layout()
  fig3.savefig(os.path.join(out_folder, 'component_values.png'))

  fig4 = plot_single_objective(
      predicted_scores, dict(o1="Predicted Scores"), axis_bound=None, c=labels,
      alpha=0.8
  )
  fig4.tight_layout()
  fig4.savefig(os.path.join(out_folder, 'scores_predicted.png'))

  fig5, ax = plt.subplots(1, 1, figsize=(subfigsz, subfigsz))
  scatter = ax.scatter(
      scores.reshape(-1), predicted_scores.reshape(-1), c=labels, alpha=0.8,
      s=sz
  )
  legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
  ax.add_artist(legend1)
  ax.set_ylabel("Predicted Scores", fontsize=fsz)
  ax.set_xlabel("Scores", fontsize=fsz)
  ax.set_xlim(scores_bound[0], scores_bound[1])
  fig5.tight_layout()
  fig5.savefig(os.path.join(out_folder, 'scores_comparison.png'))
  plt.close('all')
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", help="data path", type=str)
  parser.add_argument("-o", "--out_folder", help="output folder", type=str)
  args = parser.parse_args()
  main(args)
