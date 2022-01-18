import numpy as np
import copy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def get_cluster(data, kmax=10, get_sil_curve=False):
  sil_max = float("-inf")
  if get_sil_curve:
    # sil = PriorityQueue()
    sil = []

  # find silhouette score for each number of cluster
  for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters=k).fit(data)
    labels = kmeans.labels_
    sil_vec = silhouette_samples(data, labels, metric='euclidean')
    sil_cur = np.mean(sil_vec)
    if sil_cur > sil_max:
      sil_max = sil_cur
      sil_max_idx = k
      sil_vec_max = sil_vec.copy()
      kmeans_best = copy.deepcopy(kmeans)
    if get_sil_curve:
      # sil.put((-sil_cur, k))
      sil.append(sil_cur)

  if get_sil_curve:
    return kmeans_best, sil_vec_max, sil_max_idx, sil
  else:
    return kmeans_best, sil_vec_max, sil_max_idx


if __name__ == "__main__":
  from sklearn.datasets import make_blobs
  import matplotlib.pyplot as plt

  # Create dataset with 3 random cluster centers and 1000 datapoints
  x, y = make_blobs(
      n_samples=1000, centers=3, n_features=2, shuffle=True, random_state=31
  )
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.scatter(x[:, 0], x[:, 1], c=y)
  plt.show()

  get_cluster(x, kmax=10)
