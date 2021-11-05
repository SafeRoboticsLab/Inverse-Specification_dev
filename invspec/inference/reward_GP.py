# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# GP part is modified from:
# https://github.com/Stanford-ILIAD/active-preference-based-gpr

# We use Laplace approximation to locally approximate the probability
# p(GP_values|queries, feedback). The energy function is formulated by negative
# log of the unnormalized posterior, i.e.,
# E(f) = - log p(feedback, GP_values | queries)
#      = - log p(feedback | GP_values) - log p(GP_values | queries)
# We use Taylor series expansion around the mode (the lowest energy state)
# The Gradient of E(f) is - nabla log p(feedback | GP_values) + K^{-1} f
# The Hessian of E(f) is W + K^{-1}
# Then, we can apply Newton's method to find `fmode`.
# Finally, p(GP_values|queries, feedback) ~= q(GP_values|queries, feedback),
# q(GP_values|queries, feedback) = N(fmode, (W + K^{-1})^{-1})
# W is evaluated at fmode

# Important variables:
# self.K  # Covariance matrix of GP values
# self.Kinv
# self.W  # Hessian of negative log likelihood, - log p(feedback | GP_values)
# self.fmode  # mode of p(GP_values | feedback, queries)

import numpy as np


np.seterr(invalid='ignore')
# import scipy
from scipy import stats
from scipy.optimize import minimize, Bounds

from invspec.inference.inference import Inference


class RewardGP(Inference):

  def __init__(
      self, stateDim, actionDim, CONFIG, initialPoint, F_min=None, F_max=None,
      F_normalize=True, verbose=False
  ):
    assert stateDim+actionDim == len(initialPoint), \
        "fature size doesn't match"
    super().__init__(stateDim, actionDim, CONFIG, F_min, F_max, F_normalize)
    self.dim = len(initialPoint)  # number of features

    #= hyperparameter
    # kernel
    self.theta = 1 / (2 * CONFIG.HORIZONTAL_LENGTH**2)
    self.vertical_variation = CONFIG.VERTICAL_VARIATION
    self.noise_level = CONFIG.NOISE_LEVEL
    self.initialPoint = np.array(initialPoint)  # f(initial point set) = 0
    self.fmode = None
    # response function
    self.noise_probit = CONFIG.NOISE_PROBIT

  #region: == Interface with GA ==
  def _eval_query(self, F, **kwargs):
    """Evaluates a query.

    Args:
        F (np.ndarray): matrix of obejectives, of shape (2, #obj).

    Returns:
        float: information gain.
    """
    return self.get_information_gain(F)

  def _eval(self, F, **kwargs):
    """Evaluates design(s).

    Args:
        F (np.ndarray): matrix of obejectives, of shape (#designs, #obj).

    Returns:
        float: inferred human utility (used as fitness function in GA).
    """
    fitness = self.post_mean(F)
    return fitness

  #endregion

  #region: == UPDATING ==
  def initialize(self):
    pass

  def learn(self):
    """Updates parameters, K, W and fmode after receiving several feedback.

    Returns:
        scipy OptimizeResult.
    """
    self.K = self.get_K()
    self.Kinv = np.linalg.inv(self.K + np.identity(self.K.shape[0]) * 1e-8)
    res, objProgress = self.get_mode()
    self.fmode = res.x
    self.W = self.get_W()

    return res, objProgress

  # def updateHyperParameters(self):
  #     ell_init = np.sqrt(1 / (2*self.theta))
  #     sigma_n_init = self.noise_level
  #     pass
  #endregion

  #region: == GP ==
  def get_information_gain(self, Xstar):
    """
    Computes the information gain if we send this query to human. We want to
    maximize the information gain by considering the candidate pairs of designs
    in the current population.

    Args:
        Xstar (np.ndarray): a pair of designs, of shape (2, #obj).

    Returns:
        float: information gain.
    """
    if len(self.memory) == 0:
      return 0.
    else:
      Sigma = self.post_cov(Xstar)
      mui, muj = self.post_mean(Xstar)
      sigmap = np.sqrt(
          np.pi * np.log(2) / 2
      ) * self.noise_probit  # sqrt(pi sigma^2 ln2 / 2)
      g_12 = Sigma[0, 0] + Sigma[1, 1] - 2 * Sigma[0, 1]

      result1 = self.bin_ent(
          self.response((mui-muj) / (np.sqrt(2 * self.noise_probit**2 + g_12)))
      )
      result2 = sigmap / (np.sqrt(sigmap ** 2 + g_12)) * \
          np.exp(-0.5 * (mui - muj)**2 / (sigmap ** 2 + g_12))

      if np.isnan(result1 - result2):
        return 0.
      else:
        return result1 - result2

  def kernel(self, xi, xj):
    """kernel function. Currently we only support RBF type kernel.

    Args:
        xi (np.ndarray): the first design.
        xj (np.ndarray): the second design.

    Returns:
        float: kernel value.
    """
    dist_i_j = np.linalg.norm(xi - xj)**2
    dist_i_base = np.linalg.norm(xi - self.initialPoint)**2
    dist_j_base = np.linalg.norm(xj - self.initialPoint)**2
    kernel_base = np.exp(-self.theta * (dist_i_base+dist_j_base))
    kernel_base *= self.vertical_variation
    _kernel = self.vertical_variation * np.exp(-self.theta * dist_i_j)
    if dist_i_j <= 1e-8:
      _kernel += self.noise_level

    return _kernel - kernel_base

  def get_K(self):
    """Gets the kernel matrix for the GP.

    Returns:
        float ndarray: GP covariance matrix.
    """
    numPt = 2 * len(self.memory)
    q_1s, q_2s, _ = self.get_all_query_feedback()
    qs = self.squeeze_concatenate(q_1s, q_2s)

    _covK = np.empty(shape=(numPt, numPt), dtype=float)
    for i in range(numPt):
      for j in range(i, numPt):
        xi = qs[i, :]
        xj = qs[j, :]

        _covK[i, j] = self.kernel(xi, xj)
        if i != j:
          _covK[j, i] = _covK[i, j]
    return _covK

  def get_mode(self, tol=1e-8, maxIter=10000, warmup=False):
    """
    We use Laplace approximation to locally approximate the probability
    p(GP_values|queries, feedback). We use Taylor series expansion around the
    mode (the lowest energy state). This function helps find the mode.

    Args:
        tol (float, optional): tolerance for termination. Defaults to 1e-8.
        maxIter (int, optional): maximum #iterations. Defaults to 10000.

    Returns:
        scipy OptimizeResult.
    """

    def obj(GP_values, feedbacks, Kinv):
      numQueries = int(len(GP_values) / 2)
      diffVec = GP_values[:numQueries] - GP_values[numQueries:]
      ys = np.multiply(diffVec / self.noise_probit, feedbacks)
      phiVec = stats.norm.cdf(ys)
      log_likelihood = np.sum(np.log(phiVec))

      f_tmp = GP_values.reshape(-1, 1)
      log_prior = -0.5 * np.matmul(GP_values, np.matmul(Kinv, f_tmp))
      return (-log_likelihood - log_prior) / numQueries

    def jac(GP_values, feedbacks, Kinv):
      numQueries = int(len(GP_values) / 2)
      _jac = np.zeros(2 * numQueries)
      for i in range(numQueries):
        kappa = feedbacks[i] / self.noise_probit
        diff = GP_values[i] - GP_values[i + numQueries]
        y = kappa * diff
        _g = self.dot_response(y) / self.response(y) * kappa
        _jac[i] = _g
        _jac[i + numQueries] = -_g
      return (-_jac + np.matmul(Kinv, GP_values)) / numQueries
      # return (-_jac + np.matmul(Kinv, GP_values))

    objProgress = []

    def callback(x):
      objProgress.append(obj(x, feedbacks, Kinv))

    numQueries = len(self.memory)
    _, _, feedbacks = self.get_all_query_feedback()
    Kinv = self.Kinv

    #= Start Optimization
    x0 = np.zeros(2 * numQueries)
    if self.fmode is not None and warmup:
      half_length = int(len(self.fmode) / 2)
      x0[:half_length] = self.fmode[:half_length]
      x0[numQueries:numQueries + half_length] = self.fmode[half_length:]
    options = {}
    options['maxiter'] = maxIter
    options['disp'] = False
    bounds = Bounds(np.zeros(2 * numQueries), np.ones(2 * numQueries))
    res = minimize(
        obj, x0=x0, jac=jac, bounds=bounds, callback=callback,
        args=(feedbacks, Kinv), tol=tol, options=options
    )
    return res, objProgress

  def get_W(self):
    """
    Gets the hessian of negative log likelihood, -log p(feedback | GP_values).

    Returns:
        np.ndarray: float, the hessian of negative log likelihood.
    """
    _, _, feedbacks = self.get_all_query_feedback()
    GP_values = self.fmode
    numQueries = int(len(GP_values) / 2)
    _W = np.zeros((2 * numQueries, 2 * numQueries))
    for i in range(numQueries):
      kappa = feedbacks[i] / self.noise_probit
      diff = GP_values[i] - GP_values[i + numQueries]
      y = kappa * diff
      _h = self.ddot_response(y) * self.response(y) - self.dot_response(y)**2
      h = _h / ((self.noise_probit * self.response(y))**2)
      _W[i, i] = -h
      _W[i, i + numQueries] = h
      _W[i + numQueries, i] = h
      _W[i + numQueries, i + numQueries] = -h
    return _W

  def kstar(self, Xstar):
    """
    Gets matrix of the covariances evaluated at all pairs of training, X, and
    test points, (xi, xj).

    Args:
        Xstar (np.ndarray): array of designs and it is of shape (N_*, #obj).

    Returns:
        np.ndarray: K_* of the shape=(2*N, Nstar).
    """
    numTrain = 2 * len(self.memory)
    numTest = Xstar.shape[0]
    q_1s, q_2s, _ = self.get_all_query_feedback()
    qs = self.squeeze_concatenate(q_1s, q_2s)
    _kstar = np.empty(shape=(numTrain, numTest), dtype=float)
    for n in range(numTrain):
      xn = qs[n, :]
      for m in range(numTest):
        _kstar[n, m] = self.kernel(Xstar[m, :], xn)
    return _kstar

  def kstarstar(self, Xstar):
    """
    Gets matrix of the covariances evaluated at all pairs of test points,
    (xi, xj).

    Args:
        Xstar (np.ndarray): array of designs and it is of shape (N_*, #obj).

    Returns:
        np.ndarray: K_** of the shape=(N_*, N_*).
    """
    numTest = Xstar.shape[0]
    _kstarstar = np.empty(shape=(numTest, numTest), dtype=float)
    for i in range(numTest):
      for j in range(i, numTest):
        xi = Xstar[i, :]
        xj = Xstar[j, :]

        _kstarstar[i, j] = self.kernel(xi, xj)
        if i != j:
          _kstarstar[j, i] = _kstarstar[i, j]
    return _kstarstar

  def post_mean(self, Xstar):
    """Gets the posterior mean ~= K_*.T x K^-1 x fmode.

    Args:
        Xstar (np.ndarray): array of designs and it is of shape (N_*, #obj).

    Returns:
        np.ndarray: float, posterior mean.
    """
    if not isinstance(Xstar, np.ndarray):
      Xstar = np.array(Xstar)
    if Xstar.ndim == 1:
      Xstar = Xstar.reshape(1, -1)
    _kstar = self.kstar(Xstar)
    a = np.matmul(self.Kinv, self.fmode)
    return np.matmul(_kstar.T, a)

  def post_cov(self, Xstar):
    """Gets the posterior variance ~= K_** - K_*.T x (K+W^-1)^-1 x K_*.

    Args:
        Xstar (np.ndarray): array of designs and it is of shape (N_*, #obj).

    Returns:
        np.ndarray: float, posterior variance.
    """
    if not isinstance(Xstar, np.ndarray):
      Xstar = np.array(Xstar)
    if Xstar.ndim == 1:
      Xstar = Xstar.reshape(1, -1)
    _kstar = self.kstar(Xstar)
    _kstarstar = self.kstarstar(Xstar)
    W = self.W
    K = self.K
    # compute (K + W^-1)^-1
    tmp = np.identity(K.shape[0]) + np.matmul(W, K)
    mtx_inv = np.linalg.inv(tmp)
    return _kstarstar - np.matmul(_kstar.T, np.matmul(mtx_inv, _kstar))

  def post_mean1pt(self, x):
    """Gets the posterior mean of a single point.

    Args:
        x (np.ndarray): a design.

    Returns:
        float: posterior mean.
    """
    return self.post_mean(x)[0]

  def post_cov1pt(self, x):
    """Gets the posterior variance of a single point.

    Args:
        x (np.ndarray): a design.

    Returns:
        float: posterior variance.
    """
    return self.post_cov(x)[0, 0]

  #endregion

  #region: == utils ==
  @staticmethod
  def squeeze_concatenate(q_1s, q_2s):
    """
    Since we might want to extend to infer human preference from a trajectory,
    the design has an extra axis. We need to squeeze matrices of queries. Then,
    we concatenate them along the first axis.

    Args:
        q_1s (np.ndarray): float, matrix consists of the first design of each
            query and has the shape (#queries, #steps, #featureDim).
        q_2s (np.ndarray): float, matrix consists of the second design of
            each query and has the shape (#queries, #steps, #featureDim).

    Returns:
        np.ndarray: float, concatenated designs.
    """
    if q_1s.shape[1] == 1 and q_1s.ndim == 3:
      q_1s = np.squeeze(q_1s, axis=1)
      q_2s = np.squeeze(q_2s, axis=1)

    return np.concatenate((q_1s, q_2s), axis=0)

  @staticmethod
  def response(x):
    """
    Ssquashes its argument into the range [0, 1]. We choose the cumulative
    density function of a standard normal distribution. This is also known as
    the probit regression.

    Args:
        x (float): noise which we assume to be R.V. of N(0, 1).

    Returns:
        float: in this case, Phi(x).
    """
    return stats.norm.cdf(x)

  @staticmethod
  def dot_response(x):
    """The first derivative of the response function.

    Args:
        x (float): noise which we assume to be R.V. of N(0, 1).

    Returns:
        float: the first derivative, in this case, N(x).
    """
    return stats.norm.pdf(x)

  @staticmethod
  def ddot_response(x):
    """The second derivative of the response function.

    Args:
        x (float): noise which we assume to be R.V. of N(0, 1).

    Returns:
        float: the second derivative, in this case, -x * N(x).
    """
    return (-x) * stats.norm.pdf(x)

  @staticmethod
  def bin_ent(p):  #
    """Binary entropy function.

    Args:
        p (float): probability.

    Returns:
        float: binary entropy.
    """
    return -p * np.log2(p + 1e-8) - (1-p) * np.log2(1 - p + 1e-8)

  #endregion
