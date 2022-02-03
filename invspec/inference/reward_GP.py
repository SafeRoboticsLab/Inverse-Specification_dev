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


def logistic(x, coeff):
  """
  Equation:
      f(x, coeff) = 1 / (1 + exp(-x/coeff))

  Args:
      x (np.ndarray): random variable.
      coeff (float): 1/coeff is the logistic growth rate of the curve.

  Returns:
      np.ndarray
  """
  return 1 / (1 + np.exp(-x / coeff))


class RewardGP(Inference):

  def __init__(
      self, state_dim, action_dim, CONFIG, initial_point, input_min=None,
      input_max=None, input_normalize=True, pop_extract_type='F', verbose=False
  ):
    assert state_dim+action_dim == len(initial_point), \
        "fature size doesn't match"
    super().__init__(
        state_dim, action_dim, CONFIG, input_min, input_max, input_normalize,
        pop_extract_type
    )
    self.dim = len(initial_point)  # number of features

    #= hyperparameter
    # kernel
    self.theta = 1 / (2 * CONFIG.HORIZONTAL_LENGTH**2)
    self.vertical_variation = CONFIG.VERTICAL_VARIATION
    self.noise_level = CONFIG.NOISE_LEVEL
    self.initial_point = np.array(initial_point)  # f(initial point set) = 0
    self.fmode = None
    # response function: noise_probit = 1/confidence_coeff
    self.confidence_coeff = CONFIG.BETA
    self.mode = 'Boltzmann'
    # self.mode = 'probit'
    assert self.mode == 'Boltzmann' or self.mode == 'probit',\
        "unsupported mode!"

  #region: == Interface with GA ==
  def _eval_query(self, input, **kwargs):
    """Evaluates a query.

    Args:
        input (np.ndarray): matrix of inputs, of shape (2, self.dim).

    Returns:
        float: information gain.
    """
    return self.get_information_gain(input)

  def _eval(self, input, **kwargs):
    """Evaluates design(s).

    Args:
        input (np.ndarray): matrix of inputs, of shape (#designs, self.dim).

    Returns:
        float: inferred human utility (used as fitness function in GA).
    """
    fitness = self.post_mean(input)
    return fitness

  #endregion

  #region: == UPDATING ==
  def initialize(self):
    pass

  def learn(self):
    """Updates parameters, K, W and fmode after receiving several feedback.

    Returns:
        scipy OptimizeResult.
        list: training progress (objective values).
    """
    self.K = self.get_K()
    self.Kinv = np.linalg.inv(self.K + np.identity(self.K.shape[0]) * 1e-8)
    res, train_progress = self.get_mode()
    self.fmode = res.x
    self.W = self.get_W()

    return res, train_progress

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
      delta_mu = mui - muj

      c_1 = Sigma[0, 0] + Sigma[1, 1] - 2 * Sigma[0, 1]

      if self.mode == 'Boltzmann':
        coeff = 1 / (self.confidence_coeff**2) + np.pi / 8 * c_1
        result1 = self.bin_ent(logistic(x=delta_mu, coeff=coeff))

        tmp = 4 / (self.confidence_coeff**2) * np.log(2)
        c_2 = (
            np.exp(-0.5 * delta_mu**2 /
                   (tmp+c_1)) * np.sqrt(tmp) / np.sqrt(tmp + c_1)
        )
      else:
        noise_probit = 1 / self.confidence_coeff
        coeff = np.sqrt(2 * noise_probit**2 + c_1)
        result1 = stats.norm.cdf(delta_mu / coeff)

        tmp = np.pi * np.log(2) * noise_probit**2
        tmp_1 = tmp + 2*c_1
        tmp_2 = np.sqrt(tmp / tmp_1)
        c_2 = np.exp(-delta_mu**2 / tmp_1) * tmp_2

      if np.isnan(result1 - c_2):
        return 0.
      else:
        return result1 - c_2

  def kernel(self, xi, xj):
    """kernel function. Currently we only support RBF type kernel.

    Args:
        xi (np.ndarray): the first design.
        xj (np.ndarray): the second design.

    Returns:
        float: kernel value.
    """
    dist_i_j = np.linalg.norm(xi - xj)**2
    dist_i_base = np.linalg.norm(xi - self.initial_point)**2
    dist_j_base = np.linalg.norm(xj - self.initial_point)**2
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
        list: training progress (objective values).
    """

    def obj(GP_values, feedbacks, Kinv):
      num_queries = int(len(GP_values) / 2)
      delta_rwd = GP_values[:num_queries] - GP_values[num_queries:]
      ys, _ = self.get_rv(delta_rwd, feedbacks, self.confidence_coeff)
      likelihood_vec = self.response(ys)
      log_likelihood = np.sum(np.log(likelihood_vec))

      f_tmp = GP_values.reshape(-1, 1)
      log_prior = -0.5 * np.matmul(GP_values, np.matmul(Kinv, f_tmp))
      return (-log_likelihood - log_prior) / num_queries

    def jac(GP_values, feedbacks, Kinv):
      num_queries = int(len(GP_values) / 2)
      _jac = np.zeros(2 * num_queries)
      for i in range(num_queries):
        delta_rwd = GP_values[i] - GP_values[i + num_queries]
        y, kappa = self.get_rv(delta_rwd, feedbacks[i], self.confidence_coeff)
        _g = self.dot_response(y) / self.response(y) * kappa
        _jac[i] = _g
        _jac[i + num_queries] = -_g
      return (-_jac + np.matmul(Kinv, GP_values)) / num_queries
      # return (-_jac + np.matmul(Kinv, GP_values))

    train_progress = []

    def callback(x):
      train_progress.append(obj(x, feedbacks, Kinv))

    num_queries = len(self.memory)
    _, _, feedbacks = self.get_all_query_feedback()
    Kinv = self.Kinv

    #= Start Optimization
    x0 = np.zeros(2 * num_queries)
    if self.fmode is not None and warmup:
      half_length = int(len(self.fmode) / 2)
      x0[:half_length] = self.fmode[:half_length]
      x0[num_queries:num_queries + half_length] = self.fmode[half_length:]
    options = {}
    options['maxiter'] = maxIter
    options['disp'] = False
    bounds = Bounds(np.zeros(2 * num_queries), np.ones(2 * num_queries))
    res = minimize(
        obj, x0=x0, jac=jac, bounds=bounds, callback=callback,
        args=(feedbacks, Kinv), tol=tol, options=options
    )
    return res, train_progress

  def get_W(self):
    """
    Gets the hessian of negative log likelihood, -log p(feedback | GP_values).

    Returns:
        np.ndarray: float, the hessian of negative log likelihood.
    """
    _, _, feedbacks = self.get_all_query_feedback()
    GP_values = self.fmode
    num_queries = int(len(GP_values) / 2)
    _W = np.zeros((2 * num_queries, 2 * num_queries))
    for i in range(num_queries):
      delta_rwd = GP_values[i] - GP_values[i + num_queries]
      y, kappa = self.get_rv(delta_rwd, feedbacks[i], self.confidence_coeff)
      _h = self.ddot_response(y) * self.response(y) - self.dot_response(y)**2
      h = _h * (kappa**2) / (self.response(y)**2)
      _W[i, i] = -h
      _W[i, i + num_queries] = h
      _W[i + num_queries, i] = h
      _W[i + num_queries, i + num_queries] = -h
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

  def get_rv(self, delta_rwd, feedback, confidence_coeff):
    """
    We provide two likelihood models:
        (1) the cumulative density function of a standard normal distribution.
            This is also known as the probit regression.
        (2) Boltzmann noisy rationality, which can be rewritten as a logistic
            function.

    Args:
        delta_rwd (np.ndarray or float): reward difference.
        feedback (np.ndarray or float): human feedback.
        confidence_coeff (float): confidence coefficient. noise of probit model
            is equal to 1/coeff.

    Returns:
        np.ndarray or float: the variable for the response function.
        float: coefficient of the reward difference.
    """
    if self.mode == "Boltzmann":
      kappa = feedback * confidence_coeff
    else:
      kappa = feedback / (np.sqrt(2) * confidence_coeff)
    return kappa * delta_rwd, kappa

  def response(self, y):
    """
    Squashes its argument into the range [0, 1]. We provide two options.
        (1) the cumulative density function of a standard normal distribution.
            This is also known as the probit regression.
        (2) Boltzmann noisy rationality, which can be rewritten as a logistic
            function.

    Args:
        y (np.ndarray or float): observed variables for the response function,
            which is equal to the delta_rwd times coefficient.

    Returns:
        np.ndarray or float: the likelihood of this random variable.
    """

    if self.mode == "Boltzmann":
      return logistic(y, 1)
    else:
      return stats.norm.cdf(y)

  def dot_response(self, y):
    """The first derivative of the response function.

    Args:
        y (np.ndarray or float): observed variables for the response function,
            which is equal to the delta_rwd times coefficient.

    Returns:
        np.ndarray or float: the first derivative.
    """
    if self.mode == "Boltzmann":
      return logistic(y, 1) * (1 - logistic(y, 1))
    else:
      return stats.norm.pdf(y)

  def ddot_response(self, y):
    """The second derivative of the response function.

    Args:
        y (np.ndarray or float): observed variables for the response function,
            which is equal to the delta_rwd times coefficient.

    Returns:
        float: the second derivative.
    """
    if self.mode == "Boltzmann":
      logit = logistic(y, 1)
      return logit * (1-logit) * (1 - 2*logit)
    else:
      return (-y) * stats.norm.pdf(y)

  @staticmethod
  def bin_ent(p):
    """Binary entropy function.

    Args:
        p (float): probability.

    Returns:
        float: binary entropy.
    """
    return -p * np.log2(p + 1e-8) - (1-p) * np.log2(1 - p + 1e-8)

  #endregion
