import random

import numpy as np
from deap import algorithms, base, creator, tools

from config import Config
from model.interface import Constraint


class GASolver:
    def __init__(self, config: Config, dataset: list):
        self.num_people = len(dataset)
        self.num_groups = config.num_groups
        self.mutation_rate = config.mutation_rate
        self.num_generations = config.num_generations
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
        Finds the optimal solution using a Genetic Algorithm.

        Args:
            num_generations: The total number of generations to run.
            pop_size: The size of the population.
            cxpb: The probability of a crossover between two individuals.
            mutpb: The probability of a mutation.
        """
        print("🧪 Optimizing...")

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
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize the population.
        population = toolbox.population(n=pop_size)

        # Set up statistics tracking.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Execute the genetic algorithm.
        result, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=num_generations,
            stats=stats,
            verbose=True,
        )

        best_solution = tools.selBest(result, k=1)[0]
        best_fitness = best_solution.fitness.values[0]

        return best_solution, best_fitness, logbook
