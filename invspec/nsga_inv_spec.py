import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.population import Population

# TODO: 1. sampling instead of deterministic selection
# TODO: 2. warmup by standard GA
# TODO: 3. duplicate elimination by objectives


def binary_tournament(pop, P, algorithm, **kwargs):
  BETA = algorithm.beta
  BASE_PROB_COEFF = algorithm.base_prob_coeff
  if P.shape[1] != 2:
    raise ValueError("Only implemented for binary tournament!")

  tournament_type = algorithm.tournament_type
  S = np.full(P.shape[0], np.nan)

  for i in range(P.shape[0]):

    a, b = P[i, 0], P[i, 1]

    # if at least one solution is infeasible
    if pop[a].CV > 0.0 or pop[b].CV > 0.0:
      S[i] = compare(
          a, pop[a].CV, b, pop[b].CV, method='smaller_is_better',
          return_random_if_equal=True
      )

    # both solutions are feasible
    else:

      if tournament_type == 'comp_by_dom_and_fitness':
        rel = Dominator.get_relation(pop[a].F, pop[b].F)
        if rel == 1:
          S[i] = a
        elif rel == -1:
          S[i] = b

      elif tournament_type == 'comp_by_rank_and_fitness':
        S[i] = compare(
            a, pop[a].get("rank"), b, pop[b].get("rank"),
            method='smaller_is_better'
        )

      else:
        raise Exception("Unknown tournament type.")

      # if rank or domination relation didn't make a decision compare by
      # human fitness
      if np.isnan(S[i]):
        # S[i] = compare(a, pop[a].get("fitness"), b,
        #     pop[b].get("fitness"), method='larger_is_better',
        #     return_random_if_equal=True)
        metric = pop[P[i]].get('fitness')
        prob_un = np.exp(BETA * metric)
        prob_fitness = prob_un / np.sum(prob_un)
        prob_basis = 1 / 2
        prob = prob_fitness * (1-BASE_PROB_COEFF) + prob_basis*BASE_PROB_COEFF
        S[i] = np.random.choice(2, size=1, replace=False, p=prob)

  return S[:, None].astype(int, copy=False)


class RankAndHumanFitnessSurvival(Survival):

  def __init__(self) -> None:
    super().__init__(filter_infeasible=True)
    self.nds = NonDominatedSorting()

  def _do(self, problem, pop, n_survive, D=None, **kwargs):
    algorithm = kwargs.get('algorithm')
    BETA = algorithm.beta
    BASE_PROB_COEFF = algorithm.base_prob_coeff
    survival_type = kwargs.get("survival_type")

    # get the objective space values and objects
    F = pop.get("F").astype(float, copy=False)

    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

      # save rank in the individual class
      for _, i in enumerate(front):
        pop[i].set("rank", k)

      # current front sorted by fitness if splitting
      if len(survivors) + len(front) > n_survive:
        n_left = n_survive - len(survivors)
        if survival_type == 'crowd':
          metric = calc_crowding_distance(F[front, :])
          for j, i in enumerate(front):
            pop[i].set("crowding", metric[j])
          I = randomized_argsort(metric, order='descending')
          I = I[:n_left]
        else:
          metric = pop[front].get('fitness')
          if survival_type == 'stoc':
            prob_un = np.exp(BETA * metric)
            prob = prob_un / np.sum(prob_un)
            I = np.random.choice(
                len(front), size=n_left, replace=False, p=prob
            )
          elif survival_type == 'noisy_stoc':
            prob_un = np.exp(BETA * metric)
            prob_fitness = prob_un / np.sum(prob_un)
            prob_basis = 1 / len(front)
            prob = (
                prob_fitness * (1-BASE_PROB_COEFF) + prob_basis*BASE_PROB_COEFF
            )
            I = np.random.choice(
                len(front), size=n_left, replace=False, p=prob
            )
          elif survival_type == 'det':
            I = randomized_argsort(metric, order='descending')
            I = I[:n_left]
          else:
            raise ValueError('Unsupported survival type')

      # otherwise take the whole front unsorted
      else:
        I = np.arange(len(front))

      # extend the survivors by all or selected individuals
      survivors.extend(front[I])

    return pop[survivors]


class NSGAInvSpec(GeneticAlgorithm):

  def __init__(
      self,
      pop_size=100,
      sampling=FloatRandomSampling(),
      selection=TournamentSelection(func_comp=binary_tournament),
      crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
      mutation=PolynomialMutation(prob=None, eta=20),
      eliminate_duplicates=True,
      n_offsprings=None,
      display=MultiObjectiveDisplay(),
      warmup=0,
      survival=RankAndHumanFitnessSurvival(),
      survival_type='stoc',
      tournament_type=None,
      beta=10.,
      base_prob_coeff=0.5,
      **kwargs,
  ):
    self.beta = beta
    self.base_prob_coeff = base_prob_coeff

    super().__init__(
        pop_size=pop_size, sampling=sampling, selection=selection,
        crossover=crossover, mutation=mutation, survival=survival,
        eliminate_duplicates=eliminate_duplicates, n_offsprings=n_offsprings,
        display=display, **kwargs
    )

    if tournament_type is None:
      self.tournament_type = 'comp_by_dom_and_fitness'
    else:
      self.tournament_type = tournament_type
    self.warmup = warmup
    self.survival_type = survival_type

  def _set_optimum(self, **kwargs):
    if not has_feasible(self.pop):
      self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
    else:
      self.opt = self.pop[self.pop.get("rank") == 0]

  def next(self):
    """
    A main sub-routine in the optimization process.
    """
    # get the infill solutions
    infills = self.infill()

    # evaluate the solutions
    assert infills is not None, "solutions cannot be None"
    self.evaluator.eval(self.problem, infills, algorithm=self)
    self.advance(infills=infills)

  def _initialize_infill(self):
    """Gets the initial infills.

    Returns:
        Population: the population of this generation.
    """
    pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
    pop.set("n_gen", self.n_gen)
    return pop

  def _initialize_advance(self, infills=None, **kwargs):
    """
    Eliminates the initial population based on the survivial and assign the
    initial fitness scores.

    Args:
        infills ([type], optional): [description]. Defaults to None.
    """
    self.pop = self.survival.do(
        self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
        survival_type='crowd'
    )
    for _, ind in enumerate(self.pop):
      ind.set("fitness", 1.)

  def _advance(self, infills, **kwargs):
    """
    Eliminates the population based on the survivial and assign the fitness
    scores.

    Args:
        infills ([type]): [description]
    """
    # merge the offsprings with the current population
    if infills is not None:
      self.pop = Population.merge(self.pop, infills)

    # execute the survival to find the fittest solutions
    if self.n_gen <= self.warmup:
      self.pop = self.survival.do(
          self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
          survival_type='crowd'
      )
      for _, ind in enumerate(self.pop):
        ind.set("fitness", 1.)
    else:
      #* Calculate the human fitness of the pop
      _ = self.fitness_func.eval(self.pop)
      self.pop = self.survival.do(
          self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
          survival_type=self.survival_type
      )


def calc_crowding_distance(F, filter_out_duplicates=True):
  n_points, n_obj = F.shape

  if n_points <= 2:
    return np.full(n_points, np.inf)

  else:

    if filter_out_duplicates:
      # filter out solutions which are duplicates - duplicates get a zero
      indicator = find_duplicates(F, epsilon=1e-24)
      is_unique = np.where(np.logical_not(indicator))[0]
    else:
      # set every point to be unique without checking it
      is_unique = np.arange(n_points)

    # index the unique points of the array
    _F = F[is_unique]

    # sort each column and get index
    I = np.argsort(_F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    _F = _F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - \
        np.row_stack([np.full(n_obj, -np.inf), _F])

    # calculate the norm for each objective -
    # set to NaN if all values are equal
    norm = np.max(_F, axis=0) - np.min(_F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last = dist_to_last[:-1] / norm
    dist_to_next = dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal
    # replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also
    # reorder from sorted list
    J = np.argsort(I, axis=0)
    tmp1 = dist_to_last[J, np.arange(n_obj)]
    tmp2 = dist_to_next[J, np.arange(n_obj)]
    _cd = np.sum(tmp1 + tmp2, axis=1) / n_obj

    # save the final vector which sets the crowding distance for duplicates
    # to zero to be eliminated
    crowding = np.zeros(n_points)
    crowding[is_unique] = _cd

  # crowding[np.isinf(crowding)] = 1e+14
  return crowding


parse_doc_string(NSGAInvSpec.__init__)
