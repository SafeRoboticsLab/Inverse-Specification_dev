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
from pymoo.util.misc import has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance


def binary_tournament(pop, P, algorithm, **kwargs):
  # BASE_PROB_COEFF = algorithm.base_prob_coeff
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
      if tournament_type == 'dominance':
        rel = Dominator.get_relation(pop[a].F, pop[b].F)
        if rel == 1:
          S[i] = a
        elif rel == -1:
          S[i] = b

      elif tournament_type == 'rank':
        S[i] = compare(
            a, pop[a].get("rank"), b, pop[b].get("rank"),
            method='smaller_is_better'
        )

      else:
        raise Exception("Unknown tournament type.")

      # if rank or domination relation didn't make a decision compare by
      # human fitness or crowding distance
      if np.isnan(S[i]):
        tournament_type_second = kwargs.get("tournament_type_second")
        if tournament_type_second == "fitness":
          metric = pop[P[i]].get('fitness')
          score_diff = np.clip(
              algorithm.beta * (metric[0] - metric[1]), -20, 20
          )
          prob = 1 / (1 + np.exp(score_diff))
          # prob_basis = 1 / 2
          # prob = prob_fitness * (
          #     1-BASE_PROB_COEFF
          # ) + prob_basis*BASE_PROB_COEFF
          S[i] = np.random.choice(2, size=1, replace=False, p=[prob, 1 - prob])
        else:
          S[i] = compare(
              a, pop[a].get("crowding"), b, pop[b].get("crowding"),
              method='larger_is_better', return_random_if_equal=True
          )

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
      crowding_dist = calc_crowding_distance(F[front, :])
      if survival_type == "crowding":
        main_metric = crowding_dist
      else:
        main_metric = pop[front].get('fitness')

      # save rank in the individual class
      for j, i in enumerate(front):
        pop[i].set("rank", k)
        pop[i].set("crowding", crowding_dist[j])

      # current front sorted by fitness if splitting
      if len(survivors) + len(front) > n_survive:
        n_left = n_survive - len(survivors)
        if survival_type == "crowding":
          I = randomized_argsort(main_metric, order='descending')
          I = I[:n_left]
        # elif survival_type == 'crowd_fitness':
        #   metric_fit = pop[front].get('fitness')
        #   metric_crowd = calc_crowding_distance(F[front, :])
        #   survival_type = kwargs.get("survival_coeff")
        #   I = randomized_argsort(metric, order='descending')
        #   I = I[:n_left]
        else:  # fitness only
          if survival_type == 'stoc':
            prob_un = np.exp(BETA * main_metric)
            prob = prob_un / np.sum(prob_un)
            I = np.random.choice(
                len(front), size=n_left, replace=False, p=prob
            )
          elif survival_type == 'noisy_stoc':
            prob_un = np.exp(BETA * main_metric)
            prob_fitness = prob_un / np.sum(prob_un)
            prob_basis = 1 / len(front)
            prob = (
                prob_fitness * (1-BASE_PROB_COEFF) + prob_basis*BASE_PROB_COEFF
            )
            I = np.random.choice(
                len(front), size=n_left, replace=False, p=prob
            )
          elif survival_type == 'det':
            I = randomized_argsort(main_metric, order='descending')
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
      survival_type='det',
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
      self.tournament_type = 'dominance'
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
        survival_type="crowding"
    )
    for _, ind in enumerate(self.pop):
      ind.set("fitness", 0.)

  def _infill(self):
    """
    Gets the candidate population for the next generation.

    Returns:
        [type]: [description]
    """
    # do the mating using the current population
    if self.n_gen <= self.warmup:
      tournament_type_second = "crowding"
    else:
      tournament_type_second = "fitness"
    off = self.mating.do(
        self.problem, self.pop, self.n_offsprings, algorithm=self,
        tournament_type_second=tournament_type_second
    )

    # if the mating could not generate any new offspring (duplicate elimination
    # might make that happen)
    if len(off) == 0:
      self.termination.force_termination = True
      return

    # if not the desired number of offspring could be created
    elif len(off) < self.n_offsprings:
      if self.verbose:
        print(
            "WARNING: Mating could not produce the required number of",
            "(unique) offsprings!"
        )

    return off

  def _advance(self, infills, **kwargs):
    """
    Eliminates the population based on the survivial and assign the fitness
    scores.
    """
    # merge the offsprings with the current population
    if infills is not None:
      self.pop = Population.merge(self.pop, infills)

    # execute the survival to find the fittest solutions
    if self.n_gen <= self.warmup:
      self.pop = self.survival.do(
          self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
          survival_type="crowding"
      )
      for _, ind in enumerate(self.pop):
        ind.set("fitness", 0.)
    else:
      #* Calculate the human fitness of the pop
      fitness = self.fitness_func.eval(self.pop)
      for ind, fit in zip(self.pop, fitness):
        ind.set("fitness", fit)
      self.pop = self.survival.do(
          self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
          survival_type=self.survival_type
      )


parse_doc_string(NSGAInvSpec.__init__)
