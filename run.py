import random

import numpy as np
from tqdm import tqdm

from analyze import run_analyzer
from config import Config
from dataset import load_dataset, save_solution
from model.constraints import AgeInterval, Disjoint, GroupSizeBalance

# from model.ga import GASolver # eaSimple algorithm
from model.ga_mu_plus_lambda import GAMuPlusLambdaSolver
from summary import exp_summary


def set_seed(init_num: int) -> None:
    random.seed(init_num)
    np.random.seed(init_num)


def experiment(config: Config) -> dict:
    dataset = load_dataset(config)

    # solver = GASolver(config, dataset) # eaSimple algorithm
    solver = GAMuPlusLambdaSolver(config, dataset)

    # Hard Constraints
    solver.add_constraint(Disjoint(penalty=config.disjoint.penalty))

    # Soft Constraints
    solver.add_constraint(
        AgeInterval(
            bonus=config.age_interval.bonus,
            penalty=config.age_interval.penalty,
            target_max_diff=config.age_interval.target_max_diff,
            absolute_max_diff=config.age_interval.absolute_max_diff,
        )
    )

    solver.add_constraint(
        GroupSizeBalance(
            penalty=config.group_size.penalty,
            hard_penalty=config.group_size.hard_penalty,
            num_groups=config.num_groups,
            num_people=len(dataset),
            min_size=config.group_size.min_size,
            max_size=config.group_size.max_size,
        )
    )

    print(
        f"\n🧬 Starting genetic algorithm... (date: {config.current_date}, seed: {config.seed})"
    )

    best_solution, best_fitness, logbook = solver.solve(
        num_generations=config.num_generations,
        pop_size=config.pop_size,
        cxpb=config.crossover_rate,
        mutpb=config.mutation_rate,
    )

    print("\n🚩 Finished!")

    save_solution(config, dataset, best_solution)

    report = run_analyzer(config, dataset, best_solution, best_fitness, logbook)

    return report


def run_experiments(config: Config) -> dict:
    print(f"------------ 🏎 Starting Experiment ------------")
    print(f"Number of runs: {config.num_experiments}")
    print(f"Generations per run: {config.num_generations}")
    print(f"Population size: {config.pop_size}")
    print("-------------------------------------------------")

    all_run_histories = []
    final_best_fitnesses = []
    all_violation_summaries = []
    all_report_summaries = []

    for _ in tqdm(range(config.num_experiments)):
        set_seed(config.seed)

        report = experiment(config)

        all_run_histories.append(report["history"])
        final_best_fitnesses.append(report["best_fitness"])
        all_violation_summaries.append(report["violation_report"])
        all_report_summaries.append(report["exp_report"])

    print("--- 🏁 Experiment Finished ---")

    result = {
        "histories": all_run_histories,
        "best_fitnesses": final_best_fitnesses,
        "violation_summaries": all_violation_summaries,
        "report_summaries": all_report_summaries,
    }

    return result


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)

    result = run_experiments(config)

    exp_summary(config, result)
