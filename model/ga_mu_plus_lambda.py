import random

import numpy as np
from deap import algorithms, base, creator, tools

from config import Config
from model.interface import Constraint


class GAMuPlusLambdaSolver:
    def __init__(self, config: Config, dataset: list):
        self.num_people = len(dataset)
        self.num_groups = config.num_groups
        self.empty_group_penalty = config.empty_group_penalty

        self.dataset = dataset
        self.constraints = []

        # DEAP setup
        # 1. Fitness: Maximization problem aiming for a single objective (scalar value).
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # 2. Individual(gene): A list that uses FitnessMax.
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def add_constraint(self, constraint: Constraint) -> None:
        """Adds a constraint to the solver."""
        # print(f"\t➕ Added constraint: '{type(constraint).__name__}'")
        self.constraints.append(constraint)

    def _evaluate_fitness(self, individual: list) -> tuple:
        """
        Fitness function for DEAP.
        DEAP requires the fitness value to be a tuple (e.g., (score,)).
        Return type must be a tuple!
        """
        groups = [[] for _ in range(self.num_groups)]
        for person_idx, group_idx in enumerate(individual):
            groups[group_idx].append(self.dataset[person_idx])

        total_fitness = 0

        for group in groups:
            score = 0
            group_size = len(group)

            if group_size == 0:
                total_fitness -= self.empty_group_penalty
                continue

            for constraint in self.constraints:
                score = constraint.evaluate(group)
                total_fitness += score

                # print(f"\t* {type(constraint).__name__}, score: {score:.2f}")

        return (total_fitness,)

    def solve(
        self,
        num_generations: int,
        pop_size: int,
        cxpb: float,
        mutpb: float,
    ):
        """
        Finds the optimal solution using a (μ + λ) Genetic Algorithm.

        Args:
            num_generations: The total number of generations to run.
            pop_size: The size of the parent population (μ, Mu).
            cxpb: The probability of a crossover between two individuals.
            mutpb: The probability of a mutation.
        """
        print("🧪 Optimizing using (μ + λ) strategy...")

        toolbox = base.Toolbox()

        # Operator for creating genes: integers between 0 and num_groups-1.
        toolbox.register("attr_int", random.randint, 0, self.num_groups - 1)
        # Operator for creating individuals: iterate num_people times.
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_int,
            n=self.num_people,
        )
        # Operator for creating the population.
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic operators
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate", tools.mutUniformInt, low=0, up=self.num_groups - 1, indpb=0.03
        )

        toolbox.register("select", tools.selBest)

        MU = pop_size  # The size of parents generation (μ)
        LAMBDA = (
            pop_size * 2
        )  # The size of children generation (λ) (e.g., double of parents

        # Initialize the population (μ)
        population = toolbox.population(n=MU)

        # Set up statistics tracking.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)

        result, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=MU,
            lambda_=LAMBDA,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=num_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Getting best solution from HallOfFame
        best_solution = hof[0]
        best_fitness = best_solution.fitness.values[0]

        return best_solution, best_fitness, logbook
