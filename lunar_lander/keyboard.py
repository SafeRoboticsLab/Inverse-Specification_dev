import gym
import pygame
from gym.utils.play import play

mapping = {(pygame.K_LEFT,): 1,
           (pygame.K_RIGHT,): 3,
           (pygame.K_UP,): 2,
           (pygame.K_DOWN,): 0}
play(gym.make("LunarLander-v2"), keys_to_action=mapping)
