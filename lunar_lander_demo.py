"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Dict, Union, Optional
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from IPython.display import Image
import argparse
from argparse import Namespace

from lunar_lander.lunar_lander import LunarLander, LunarLanderContinuous


def heuristic(env, s):
  """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
        a: The heuristic to be fed into the step function defined above to
            determine the next step and reward.
    """

  angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
  if angle_targ > 0.4:
    angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
  if angle_targ < -0.4:
    angle_targ = -0.4
  hover_targ = 0.55 * np.abs(
      s[0]
  )  # target y should be proportional to horizontal offset

  angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
  hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

  if s[6] or s[7]:  # legs have contact
    angle_todo = 0
    hover_todo = (
        -(s[3]) * 0.5
    )  # override to reduce fall speed, that's all we need after contact

  if env.continuous:
    a = np.array([hover_todo*20 - 1, -angle_todo * 20])
    a = np.clip(a, -1, +1)
  else:
    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
      a = 2
    elif angle_todo < -0.05:
      a = 3
    elif angle_todo > +0.05:
      a = 1
  return a


def main(args: Namespace):
  if args.continuous:
    env = LunarLanderContinuous(args)
  else:
    env = LunarLander(args)
  env.seed(args.seed)
  total_reward = 0
  steps = 0
  s = env.reset()

  if args.render and args.mode == "rgb_array":
    rgb = env.render(mode=args.mode)
    fig_folder = args.fig_folder
    os.makedirs(fig_folder, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(rgb)
    fig.savefig(os.path.join(fig_folder, f"{steps:d}.png"))
  # input()

  while True:
    a = heuristic(env, s)
    s, r, done, info = env.step(a)
    total_reward += r

    if args.render:
      rgb = env.render(mode=args.mode)
      if args.mode == "rgb_array":
        ax.clear()
        ax.imshow(rgb)
        fig.savefig(os.path.join(fig_folder, f"{steps:d}.png"))

    if steps % 20 == 0 or done:
      print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
      print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      # input()
    steps += 1
    if done:
      break
  if args.render:
    env.close()
  return total_reward


if __name__ == "__main__":
  # python lunar_lander_demo.py -c -r -s 0 -m rgb_array
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--seed", help="seed", type=int, default=None)
  parser.add_argument(
      "-m", "--mode", help="rendering mode", type=str, default="human"
  )
  parser.add_argument('-c', '--continuous', action='store_true')
  parser.add_argument('-r', '--render', action='store_true')
  parser.add_argument(
      "-f", "--fig_folder", help="rendering mode", type=str,
      default=os.path.join("figure", "demo_heu_lander", "progress")
  )
  parser.add_argument('-a', '--auxiliary', action='store_true')
  parser.add_argument(
      "-ar", "--aux_ratio", help="auxiliary side engine power ratio",
      type=float, default=0.5
  )
  args = parser.parse_args()
  main(args)
