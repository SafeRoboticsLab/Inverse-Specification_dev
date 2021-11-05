# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )


class BaseConfig():

  def __init__(
      self, SEED=0, MEMORY_CAPACITY=100, MAX_QUERIES=100, MAX_QUERIES_PER=3
  ):
    """

    Args:
        SEED (int, optional): random seed.
            Defaults to 0
        MEMORY_CAPACITY (int, optional): the size of replay buffer.
            Defaults to 100
        MAX_QUERIES (int, optional): maximal number of queries.
            Defaults to 100
        MAX_QUERIES_PER (int, optional): maximal number of queries per update.
            Defaults to 3
    """
    self.SEED = SEED
    self.MAX_QUERIES = MAX_QUERIES
    self.MEMORY_CAPACITY = MEMORY_CAPACITY
    self.MAX_QUERIES_PER = MAX_QUERIES_PER


class NNConfig(BaseConfig):

  def __init__(
      self, SEED=0, MEMORY_CAPACITY=100, MAX_QUERIES=100, MAX_QUERIES_PER=3,
      DEVICE='cpu', ARCHITECTURE=[20], ACTIVATION='Tanh', MAX_UPDATES=10000,
      BATCH_SIZE=32, MAX_MODEL=50, LR=1e-3, LR_END=1e-4, LR_PERIOD=1000,
      LR_DECAY=0.5, TRADEOFF=0.1, MAX_GRAD_NORM=1.
  ):
    """
    Args:
        DEVICE (str, optional): on which you want to run your PyTorch model.
            Defaults to 'cpu'.
        MAX_UPDATES (int, optional): maximal number of gradient updates.
            Defaults to 100000.
        LR (float, optional): learning rate.
            Defaults to 1e-3.
        LR_END (float, optional): terminal value of LR.
            Defaults to 1e-4.
        LR_PERIOD (int, optional): update period of LR.
            Defaults to 1000.
        LR_DECAY (float, optional): multiplicative factor of LR.
            Defaults to 0.5.
        BATCH_SIZE (int, optional): the number of samples used to calculate
            the approximate gradient. Defaults to 32.
        MAX_MODEL (int, optional): maximal number of models you want to
            store during the training process. Defaults to 50.
        ARCHITECTURE (list, optional): the architecture of the hidden layers
            of the NN. Defaults to [20].
        ACTIVATION (str, optional): the activation function used in the NN.
            Defaults to 'Tanh'.
        TRADEOFF (float, optional): the balance between matching human
            prediction and deviation from model concerning reward only.
            Defaults to 0.1.
        MAX_GRAD_NORM (float, optional): Maximum gradient norm.
            Defaults to 5.
    """
    super().__init__(
        SEED=SEED, MEMORY_CAPACITY=MEMORY_CAPACITY, MAX_QUERIES=MAX_QUERIES,
        MAX_QUERIES_PER=MAX_QUERIES_PER
    )

    self.MAX_UPDATES = MAX_UPDATES

    self.LR = LR
    self.LR_END = LR_END
    self.LR_PERIOD = LR_PERIOD
    self.LR_DECAY = LR_DECAY

    self.BATCH_SIZE = BATCH_SIZE

    self.MAX_MODEL = MAX_MODEL
    self.DEVICE = DEVICE

    self.ARCHITECTURE = ARCHITECTURE
    self.ACTIVATION = ACTIVATION

    self.TRADEOFF = TRADEOFF
    self.MAX_GRAD_NORM = MAX_GRAD_NORM


class GPConfig(BaseConfig):

  def __init__(
      self, SEED=0, MEMORY_CAPACITY=100, MAX_QUERIES=100, MAX_QUERIES_PER=3,
      HORIZONTAL_LENGTH=1., VERTICAL_VARIATION=1., NOISE_LEVEL=0.1,
      NOISE_PROBIT=0.05
  ):
    """

    Args:
        HORIZONTAL_LENGTH (float, optional): ell in kernel.
            Defaults to 1..
        VERTICAL_VARIATION (float, optional): sigma_f in kernel.
            Defaults to 1..
        NOISE_LEVEL (float, optional): sigma_n in kernel.
            Defaults to 0.1.
        NOISE_PROBIT (float, optional): sigma in probit regression
            Defaults to 0.05.
    """
    super().__init__(
        SEED=SEED, MEMORY_CAPACITY=MEMORY_CAPACITY, MAX_QUERIES=MAX_QUERIES,
        MAX_QUERIES_PER=MAX_QUERIES_PER
    )

    self.HORIZONTAL_LENGTH = HORIZONTAL_LENGTH
    self.VERTICAL_VARIATION = VERTICAL_VARIATION
    self.NOISE_LEVEL = NOISE_LEVEL
    self.NOISE_PROBIT = NOISE_PROBIT
