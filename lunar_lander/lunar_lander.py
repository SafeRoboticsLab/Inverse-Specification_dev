"""
Modified from OpenAI Gym's implementation (Created by Oleg Klimov).
Author: Kai-Chieh Hsu (kaichieh@princeton.edu)

Notes:
  1. Box2D uses the coordinate: +x: right and +y: up.
  2. We define the body frame: +x: right and +y: up, which means the body frame
    is exactly the same as the world frame when yaw=0. This is different from
    the gym's implementation. To make it clear, we change `tip` to `body_y` and
    `side` to `body_x`.
  3. The force uses only the offset direction (previously the force is directly
    multiplied by the offset).

--- Below are the description from the OpenAI Gym ---
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full
throttle or turn it off. That's the reason this environment is OK to have
discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first
two numbers in the state vector. Reward for moving from the top of the screen
to the landing pad and zero speed is about 100..140 points. If the lander moves
away from the landing pad it loses reward. The episode finishes if the lander
crashes or comes to rest, receiving an additional -100 or +100 points. Each leg
with ground contact is +10 points. Firing the main engine is -0.3 points each
frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can
learn to fly and then land on its first attempt. Please see the source code for
details.

To see a heuristic landing, run:
    python gym/envs/box2d/lunar_lander.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import Box2D
from Box2D.b2 import (
    edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef,
    contactListener
)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
# affects how fast-paced the game is, forces should be adjusted as well
SCALE = 30.0

MAIN_ENGINE_POWER = 6.0
SIDE_ENGINE_POWER = 0.3

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0),
               (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

# Default engine positions
MAIN_ENGINE_HEIGHT = -4.0
SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0
SIDE_ENGINE_HEIGHT_AUX = -8.0
SIDE_ENGINE_AWAY_AUX = 14.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):

  def __init__(self, env):
    contactListener.__init__(self)
    self.env = env

  def BeginContact(self, contact):
    if (
        self.env.lander == contact.fixtureA.body
        or self.env.lander == contact.fixtureB.body
    ):
      self.env.game_over = True
    for i in range(2):
      if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
        self.env.legs[i].ground_contact = True

  def EndContact(self, contact):
    for i in range(2):
      if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
        self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": FPS
  }

  continuous = False

  def __init__(self, config=None):
    EzPickle.__init__(self)
    self.main_engine_height = MAIN_ENGINE_HEIGHT / SCALE
    self.side_engine_height = SIDE_ENGINE_HEIGHT / SCALE
    self.side_engine_away = SIDE_ENGINE_AWAY / SCALE
    self.lander_poly = LANDER_POLY
    self.auxiliary = False  #TODO: adds description.
    if config is not None:
      self.main_engine_height = (
          getattr(config, "main_engine_height", MAIN_ENGINE_HEIGHT) / SCALE
      )
      self.side_engine_height = (
          getattr(config, "side_engine_height", SIDE_ENGINE_HEIGHT) / SCALE
      )
      self.side_engine_away = (
          getattr(config, "side_engine_away", SIDE_ENGINE_AWAY) / SCALE
      )
      self.lander_poly = getattr(config, "lander_poly", LANDER_POLY)
      self.auxiliary = getattr(config, "auxiliary", False)
      if self.auxiliary:
        self.side_engine_height_aux = (
            getattr(config, "side_engine_height_aux", SIDE_ENGINE_HEIGHT_AUX)
            / SCALE
        )
        self.side_engine_away_aux = (
            getattr(config, "side_engine_away_aux", SIDE_ENGINE_AWAY_AUX)
            / SCALE
        )
        self.side_engine_away_aux = (
            getattr(config, "side_engine_away_aux", SIDE_ENGINE_AWAY_AUX)
            / SCALE
        )
        self.aux_ratio = getattr(config, "aux_ratio", 0.5)

    self.seed()
    self.viewer = None

    self.world = Box2D.b2World()
    self.moon = None
    self.lander = None
    self.particles = []

    self.prev_reward = None

    # useful range is -1 .. +1, but spikes can be higher
    self.observation_space = spaces.Box(
        -np.inf, np.inf, shape=(8,), dtype=np.float32
    )

    if self.continuous:
      # Action is two floats [main engine, left-right engines].
      # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine
      # can't work with less than 50% power.
      # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine,
      #  -0.5..0.5 off
      self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
    else:
      # Nop, fire left engine, main engine, right engine
      self.action_space = spaces.Discrete(4)

    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _destroy(self):
    if not self.moon:
      return
    self.world.contactListener = None
    self._clean_particles(True)
    self.world.DestroyBody(self.moon)
    self.moon = None
    self.world.DestroyBody(self.lander)
    self.lander = None
    self.world.DestroyBody(self.legs[0])
    self.world.DestroyBody(self.legs[1])

  def reset(self):
    self._destroy()
    self.world.contactListener_keepref = ContactDetector(self)
    self.world.contactListener = self.world.contactListener_keepref
    self.game_over = False
    self.prev_shaping = None

    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE

    # terrain
    CHUNKS = 11
    height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
    chunk_x = [W / (CHUNKS-1) * i for i in range(CHUNKS)]
    self.helipad_x1 = chunk_x[CHUNKS//2 - 1]
    self.helipad_x2 = chunk_x[CHUNKS//2 + 1]
    self.helipad_y = H / 4
    height[CHUNKS//2 - 2] = self.helipad_y
    height[CHUNKS//2 - 1] = self.helipad_y
    height[CHUNKS//2 + 0] = self.helipad_y
    height[CHUNKS//2 + 1] = self.helipad_y
    height[CHUNKS//2 + 2] = self.helipad_y
    smooth_y = [
        0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
        for i in range(CHUNKS)
    ]

    self.moon = self.world.CreateStaticBody(
        shapes=edgeShape(vertices=[(0, 0), (W, 0)])
    )
    self.sky_polys = []
    for i in range(CHUNKS - 1):
      p1 = (chunk_x[i], smooth_y[i])
      p2 = (chunk_x[i + 1], smooth_y[i + 1])
      self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
      self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

    self.moon.color1 = (0.0, 0.0, 0.0)
    self.moon.color2 = (0.0, 0.0, 0.0)

    initial_y = VIEWPORT_H / SCALE
    self.lander = self.world.CreateDynamicBody(
        position=(VIEWPORT_W / SCALE / 2, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(
                vertices=[(x / SCALE, y / SCALE) for x, y in self.lander_poly]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0,
        ),  # 0.99 bouncy
    )
    self.lander.color1 = (0.5, 0.4, 0.9)
    self.lander.color2 = (0.3, 0.3, 0.5)
    self.lander.ApplyForceToCenter(
        (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        ),
        True,
    )

    self.legs = []
    for i in [-1, +1]:
      leg = self.world.CreateDynamicBody(
          position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
          angle=(i * 0.05),
          fixtures=fixtureDef(
              shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
              density=1.0,
              restitution=0.0,
              categoryBits=0x0020,
              maskBits=0x001,
          ),
      )
      leg.ground_contact = False
      leg.color1 = (0.5, 0.4, 0.9)
      leg.color2 = (0.3, 0.3, 0.5)
      rjd = revoluteJointDef(
          bodyA=self.lander,
          bodyB=leg,
          localAnchorA=(0, 0),
          localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
          enableMotor=True,
          enableLimit=True,
          maxMotorTorque=LEG_SPRING_TORQUE,
          motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
      )
      if i == -1:
        # The most esoteric numbers here, angled legs have freedom to travel
        # within
        rjd.lowerAngle = (+0.9 - 0.5)
        rjd.upperAngle = +0.9
      else:
        rjd.lowerAngle = -0.9
        rjd.upperAngle = -0.9 + 0.5
      leg.joint = self.world.CreateJoint(rjd)
      self.legs.append(leg)

    self.drawlist = [self.lander] + self.legs

    # pos = self.lander.position
    # vel = self.lander.linearVelocity
    # state = np.array([
    #     (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
    #     (pos.y - (self.helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
    #     vel.x * (VIEWPORT_W/SCALE/2) / FPS,
    #     vel.y * (VIEWPORT_H/SCALE/2) / FPS,
    #     self.lander.angle,
    #     20.0 * self.lander.angularVelocity / FPS,
    #     1.0 if self.legs[0].ground_contact else 0.0,
    #     1.0 if self.legs[1].ground_contact else 0.0,
    # ])
    # print("After reset")
    # with np.printoptions(precision=2, suppress=False):
    #   print(state)

    return self.step(np.array([0, 0]) if self.continuous else 0)[0]

  def _create_particle(self, density, x, y, ttl):
    p = self.world.CreateDynamicBody(
        position=(x, y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
            density=density,
            friction=0.1,
            categoryBits=0x0100,
            maskBits=0x001,  # collide only with ground
            restitution=0.3,
        ),
    )
    p.ttl = ttl
    self.particles.append(p)
    self._clean_particles(False)
    return p

  def _clean_particles(self, all):
    while self.particles and (all or self.particles[0].ttl < 0):
      self.world.DestroyBody(self.particles.pop(0))

  def step(self, action):
    if self.continuous:
      action = np.clip(action, -1, +1).astype(np.float32)
    else:
      assert self.action_space.contains(action), "%r (%s) invalid " % (
          action,
          type(action),
      )

    # Engines
    body_x = np.array([np.cos(self.lander.angle), np.sin(self.lander.angle)])
    body_y = np.array([-np.sin(self.lander.angle), np.cos(self.lander.angle)])
    # Adds noises in the impulse position.
    if self.auxiliary:
      dispersion = self.np_random.uniform(-1.0, +1.0, size=6) / SCALE
    else:
      dispersion = self.np_random.uniform(-1.0, +1.0, size=4) / SCALE

    # Computes the impulse: dp = F*dt
    m_power = 0.0
    if ((self.continuous and action[0] > 0.0)
        or (not self.continuous and action == 2)):
      # Main engine
      if self.continuous:
        m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
        assert m_power >= 0.5 and m_power <= 1.0
      else:
        m_power = 1.0
      m_offset = (
          self.main_engine_height * body_y + dispersion[0] * body_x
          + 2 * dispersion[1] * body_y
      )
      m_offset_dir = m_offset / np.linalg.norm(m_offset)
      impulse_pos = (
          self.lander.position[0] + m_offset[0],
          self.lander.position[1] + m_offset[1]
      )
      p = self._create_particle(
          3.5,  # 3.5 is here to make particle speed adequate
          impulse_pos[0],
          impulse_pos[1],
          m_power,
      )  # particles are just a decoration
      p.ApplyLinearImpulse(
          (
              m_offset_dir[0] * MAIN_ENGINE_POWER * m_power,
              m_offset_dir[1] * MAIN_ENGINE_POWER * m_power
          ),
          impulse_pos,
          True,
      )
      self.lander.ApplyLinearImpulse(
          (
              -m_offset_dir[0] * MAIN_ENGINE_POWER * m_power,
              -m_offset_dir[1] * MAIN_ENGINE_POWER * m_power
          ),
          impulse_pos,
          True,
      )

    s_power = 0.0
    if ((self.continuous and np.abs(action[1]) > 0.5)
        or (not self.continuous and action in [1, 3])):
      # Orientation engines
      if self.continuous:
        direction = np.sign(action[1])
        # Engine activates only if more than 50% power.
        s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
        assert s_power >= 0.5 and s_power <= 1.0
      else:
        direction = action - 2  # Maps to -1 and 1.
        s_power = 1.0
      s_offset = (
          self.side_engine_height * body_y
          + direction * self.side_engine_away * body_x
          + 3 * dispersion[2] * body_x + dispersion[3] * body_y
      )

      s_offset_dir = s_offset / np.linalg.norm(s_offset)
      impulse_pos = (
          self.lander.position[0] + s_offset[0],
          self.lander.position[1] + s_offset[1],
      )
      p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
      p.ApplyLinearImpulse(
          (
              s_offset_dir[0] * SIDE_ENGINE_POWER * s_power,
              s_offset_dir[1] * SIDE_ENGINE_POWER * s_power
          ),
          impulse_pos,
          True,
      )
      self.lander.ApplyLinearImpulse(
          (
              -s_offset_dir[0] * SIDE_ENGINE_POWER * s_power,
              -s_offset_dir[1] * SIDE_ENGINE_POWER * s_power
          ),
          impulse_pos,
          True,
      )
      if self.auxiliary:
        # -direction for diagonal
        s_offset_aux = (
            self.side_engine_height_aux * body_y
            - direction * self.side_engine_away * body_x
            + 3 * dispersion[4] * body_x + dispersion[5] * body_y
        )

        s_offset_dir_aux = s_offset_aux / np.linalg.norm(s_offset_aux)
        impulse_pos_aux = (
            self.lander.position[0] + s_offset_aux[0],
            self.lander.position[1] + s_offset_aux[1],
        )
        p_aux = self._create_particle(
            0.7, impulse_pos_aux[0], impulse_pos_aux[1],
            s_power * self.aux_ratio
        )
        p_aux.ApplyLinearImpulse(
            (
                s_offset_dir_aux[0] * SIDE_ENGINE_POWER * s_power
                * self.aux_ratio, s_offset_dir_aux[1] * SIDE_ENGINE_POWER
                * s_power * self.aux_ratio
            ),
            impulse_pos_aux,
            True,
        )
        self.lander.ApplyLinearImpulse(
            (
                -s_offset_dir_aux[0] * SIDE_ENGINE_POWER * s_power
                * self.aux_ratio, -s_offset_dir_aux[1] * SIDE_ENGINE_POWER
                * s_power * self.aux_ratio
            ),
            impulse_pos_aux,
            True,
        )

    self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    pos = self.lander.position
    vel = self.lander.linearVelocity
    state = [
        (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
        (pos.y - (self.helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
        vel.x * (VIEWPORT_W/SCALE/2) / FPS,
        vel.y * (VIEWPORT_H/SCALE/2) / FPS,
        self.lander.angle,
        20.0 * self.lander.angularVelocity / FPS,
        1.0 if self.legs[0].ground_contact else 0.0,
        1.0 if self.legs[1].ground_contact else 0.0,
    ]
    assert len(state) == 8

    reward = 0
    shaping = (
        -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]
    )  # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    if self.prev_shaping is not None:
      reward = shaping - self.prev_shaping
    self.prev_shaping = shaping

    reward -= (
        m_power * 0.30
    )  # less fuel spent is better, about -30 for heuristic landing
    reward -= s_power * 0.03

    done = False
    if self.game_over or abs(state[0]) >= 1.0:
      done = True
      reward = -100
    if not self.lander.awake:
      done = True
      reward = +100
    return np.array(state, dtype=np.float32), reward, done, {}

  def render(self, mode="human"):
    from gym.envs.classic_control import rendering

    if self.viewer is None:
      self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
      self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

    for obj in self.particles:
      obj.ttl -= 0.15
      obj.color1 = (
          max(0.2, 0.2 + obj.ttl),
          max(0.2, 0.5 * obj.ttl),
          max(0.2, 0.5 * obj.ttl),
      )
      obj.color2 = (
          max(0.2, 0.2 + obj.ttl),
          max(0.2, 0.5 * obj.ttl),
          max(0.2, 0.5 * obj.ttl),
      )

    self._clean_particles(False)

    for p in self.sky_polys:
      self.viewer.draw_polygon(p, color=(0, 0, 0))

    for obj in self.particles + self.drawlist:
      for f in obj.fixtures:
        trans = f.body.transform
        if type(f.shape) is circleShape:
          t = rendering.Transform(translation=trans * f.shape.pos)
          self.viewer.draw_circle(f.shape.radius, 20,
                                  color=obj.color1).add_attr(t)
          self.viewer.draw_circle(
              f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2
          ).add_attr(t)
        else:
          path = [trans * v for v in f.shape.vertices]
          self.viewer.draw_polygon(path, color=obj.color1)
          path.append(path[0])
          self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    for x in [self.helipad_x1, self.helipad_x2]:
      flagy1 = self.helipad_y
      flagy2 = flagy1 + 50/SCALE
      self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
      self.viewer.draw_polygon(
          [
              (x, flagy2),
              (x, flagy2 - 10/SCALE),
              (x + 25/SCALE, flagy2 - 5/SCALE),
          ],
          color=(0.8, 0.8, 0),
      )

    return self.viewer.render(return_rgb_array=mode == "rgb_array")

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None


class LunarLanderContinuous(LunarLander):
  continuous = True


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


def demo_heuristic_lander(env, seed=None, render=False, mode="human"):
  env.seed(seed)
  total_reward = 0
  steps = 0
  s = env.reset()

  if render and mode == "rgb_array":
    rgb = env.render(mode=mode)
    fig_folder = os.path.join("figure", "demo_heu_lander", "progress")
    os.makedirs(fig_folder, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(rgb)
    fig.savefig(os.path.join(fig_folder, f"{steps:d}.png"))
  # input()

  while True:
    a = heuristic(env, s)
    s, r, done, info = env.step(a)
    total_reward += r

    if render:
      rgb = env.render(mode=mode)
      if mode == "rgb_array":
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
  if render:
    env.close()
  return total_reward


if __name__ == "__main__":
  demo_heuristic_lander(LunarLander(), render=True, mode='human', seed=0)
