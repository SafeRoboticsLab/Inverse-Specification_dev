# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
#== LOSS FUNC ==
KL = torch.nn.KLDivLoss(reduction='batchmean')
MSE = torch.nn.MSELoss(reduction='mean')

from funct_approx.model import NeuralNetwork
from invspec.inference.inference import Inference


class RewardNN(Inference):

  def __init__(
      self, stateDim, actionDim, CONFIG, F_min=None, F_max=None,
      F_normalize=True, beta=10, verbose=False, boundedOutput=True
  ):

    super().__init__(stateDim, actionDim, CONFIG, F_min, F_max, F_normalize)

    #== PARAM ==
    # Learning Rate
    self.LR = CONFIG.LR
    self.LR_PERIOD = CONFIG.LR_PERIOD
    self.LR_DECAY = CONFIG.LR_DECAY
    self.LR_END = CONFIG.LR_END

    # loss function
    self.tradeoff = CONFIG.TRADEOFF
    self.max_grad_norm = CONFIG.MAX_GRAD_NORM

    # human model
    self.beta = beta

    #== NEURAL NETWORK ==
    self.dimList = [stateDim+actionDim] + CONFIG.ARCHITECTURE + [1]
    self.actType = CONFIG.ACTIVATION
    self.boundedOutput = boundedOutput
    self.device = CONFIG.DEVICE
    self.BATCH_SIZE = CONFIG.BATCH_SIZE

    self.build_NN(verbose)
    self.build_optimizer()

  #== Interface with GA ==
  def _eval(self, F, **kwargs):
    trajectories = np.expand_dims(F, axis=1)
    with torch.no_grad():
      self.reward.eval()
      fitness = self.get_reward_traj(trajectories)
    fitness = fitness.detach().cpu().numpy().reshape(-1)
    return fitness

  def _eval_query(self, F, **kwargs):
    return

  #== Function Approximator ==
  def build_NN(self, verbose=False):
    if self.boundedOutput:
      self.reward = NeuralNetwork(
          self.dimList, self.actType, verbose, torch.nn.Tanh()
      )
    else:
      self.reward = NeuralNetwork(
          self.dimList, self.actType, verbose, torch.nn.Identity()
      )

    if self.device == torch.device('cuda'):
      self.reward.cuda()

  def build_optimizer(self):
    self.optimizer = optim.AdamW(
        self.reward.parameters(), lr=self.LR, weight_decay=1e-3
    )
    self.scheduler = optim.lr_scheduler.StepLR(
        self.optimizer, step_size=self.LR_PERIOD, gamma=self.LR_DECAY
    )
    self.cntUpdate = 0

  #== RETRIEVING ==
  def get_reward_traj(self, trajectories):
    x = torch.FloatTensor(trajectories).to(self.device)
    if self.boundedOutput:  # map to [0, 1]
      reward_components = self.reward(x) * 0.5 + 0.5
    else:
      reward_components = self.reward(x)
    reward_sum = torch.sum(reward_components, axis=1)
    return reward_sum

  def get_utility_traj(self, trajectories):
    reward = self.get_reward_traj(trajectories)
    return torch.exp(self.beta * reward)

  def get_prob_without_cons(self, q_1s, q_2s):
    trajectories_1 = self.extract(q_1s)
    trajectories_2 = self.extract(q_2s)
    batch_size = trajectories_1.shape[0]

    u_1 = self.get_utility_traj(trajectories_1)
    u_2 = self.get_utility_traj(trajectories_2)
    p_1 = u_1 / (u_1+u_2)
    p_1 = p_1.reshape(-1)
    p_2 = 1 - p_1

    p = torch.empty(size=(batch_size, 2)).to(self.device)
    p[:, 0] = p_1.clone()
    p[:, 1] = p_2.clone()
    p.retain_grad()
    return p

  def predict_traj(self, trajectories):
    """Predicts the ranking of the input trajectories.

    Args:
        trajectories (np.ndarray): has shape (batch_size, length,
        stateDim+actionDim).

    Returns:
        ndarray: predicted order.
        ndarray: predicted scores.
    """
    self.reward.eval()
    trajTensor = torch.FloatTensor(trajectories).to(self.device)
    with torch.no_grad():
      if self.boundedOutput:  # map to [0, 1]
        reward_components = self.reward(trajTensor) * 0.5 + 0.5
      else:
        reward_components = self.reward(trajTensor)
      reward_sum = torch.sum(reward_components, axis=1)
      reward_sum = reward_sum.cpu().numpy().reshape(-1)

    order = np.argsort(-reward_sum)
    return order, reward_sum

  def initialize(self):
    self.reward._initialize_weights()
    del self.optimizer
    del self.scheduler
    self.build_optimizer()
    pass

  #== UPDATING ==
  def learn(self, numIter, check_period=100, initialize=False):
    torch.autograd.set_detect_anomaly(True)
    if initialize:
      print("Init!")
      self.initialize()

    #== Train ==
    loss_record = np.empty(shape=(numIter, 3))
    for it in range(numIter):
      self.reward.train()
      lr_cur = self.optimizer.state_dict()['param_groups'][0]['lr']
      #= BATCH AND TRAJECTORIES EXTRACTION
      trajectories_1, trajectories_2, fs = \
          self.get_sampled_query_feedback(self.BATCH_SIZE)

      fs = torch.FloatTensor(fs).to(self.device)
      ps = torch.empty(size=fs.shape).to(self.device)

      #= GET REWARD, FEASIBILITY AND UTILITY
      r_1 = self.get_reward_traj(trajectories_1)
      r_2 = self.get_reward_traj(trajectories_2)

      nu_1 = torch.exp(self.beta * r_1)
      nu_2 = torch.exp(self.beta * r_2)
      p_1 = nu_1 / (nu_1+nu_2)
      p_1 = p_1.reshape(-1)
      p_2 = 1 - p_1
      ps[:, 0] = p_1.clone()
      ps[:, 1] = p_2.clone()
      ps.retain_grad()
      # ps = self.getProbWithCons(batch.q_1, batch.q_2,
      #     fixReward=False, fixConstraint=True)
      log_ps = torch.log(ps + 1e-8)

      #= REGULARIZATION
      zeros = torch.zeros_like(r_1)
      reg_1 = MSE(input=r_1, target=zeros)
      reg_2 = MSE(input=r_2, target=zeros)
      reg = self.tradeoff * (reg_1+reg_2)

      #= LOSS
      clf_loss = KL(input=log_ps, target=fs)
      loss = clf_loss + reg

      self.optimizer.zero_grad()
      loss.backward()
      clip_grad_norm_(self.reward.parameters(), self.max_grad_norm)
      self.optimizer.step()

      if lr_cur <= self.LR_END:
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.LR_END
      else:
        self.scheduler.step()

      loss_value = loss.detach().cpu().numpy()
      clf_loss_value = clf_loss.detach().cpu().numpy()
      reg_value = reg.detach().cpu().numpy()

      print("[{:d}]: {:.2f}".format(it, loss_value), end='\r')
      if (it+1) % check_period == 0:
        print("[{:d} / {:d}]: ".format(it, numIter), end='')
        print(
            "(loss, clf, reg) = ({:.2f}, {:.2f}, {:.2f}) ".format(
                loss_value, clf_loss_value, reg_value
            ), end=''
        )
        print("with learning rate = {:.2e}".format(lr_cur))

      loss_record[it, 0] = loss_value
      loss_record[it, 1] = clf_loss_value
      loss_record[it, 2] = reg_value
    return loss_record
