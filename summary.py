import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config


def exp_summary(
    config: Config,
    result: dict,
) -> None:
    exp_results_filename = f"./result/experiment_results_{config.run_name}.json"

    save_data = {
        "experiment_id": f"exp_{config.current_date}",
        "base_seed": config.seed,
        "num_experiments": config.num_experiments,
        "config": {
            "num_generations": config.num_generations,
            "pop_size": config.pop_size,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
        },
        "final_best_fitnesses": result["best_fitnesses"],
        "all_run_histories_max": [h["max_fitness"] for h in result["histories"]],
        "all_violation_summaries": result["violation_summaries"],
        "all_report_summaries": result["report_summaries"],
    }

    try:
        with open(exp_results_filename, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Results data saved to {exp_results_filename}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    figure_filename = f"./result/experiment_plot_{config.run_name}.png"
    try:
        plot_experiment_results(
            result["histories"],
            result["best_fitnesses"],
            result["violation_summaries"],
            result["report_summaries"],
            figure_filename,
        )
        print(f"Performance plot saved to {figure_filename}")
    except Exception as e:
        print(f"Error generating plot: {e}")

    print("--- Experiments summary done ---")


def plot_experiment_results(
    all_run_histories: list,
    final_best_fitnesses: list,
    all_violation_summaries: list,
    all_report_summaries: list,
    figure_filename: str,
) -> None:
    """Creates and saves a 2x2 plot of the experiment results."""

    print(f"Generating plot and saving to {figure_filename}...")

    plt.figure(figsize=(20, 16))

    # Convert summary lists to DataFrames for easier plotting
    violation_df = pd.DataFrame(all_violation_summaries)
    report_df = pd.DataFrame(all_report_summaries)

    # --- Plot 1: Histogram of Final Scores ---
    plt.subplot(2, 2, 1)

    plt.hist(final_best_fitnesses, bins=15, edgecolor="black", alpha=0.7, color="green")

    mean_score = np.mean(final_best_fitnesses)
    median_score = np.median(final_best_fitnesses)
    plt.axvline(
        mean_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.2f}",
    )
    plt.axvline(
        median_score,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_score:.2f}",
    )
    plt.legend()

    plt.title(
        f"Distribution of Final Fitness Scores (N={len(final_best_fitnesses)})",
        fontsize=14,
    )
    plt.xlabel("Final Fitness Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Plot 2: Convergence Curves ---
    plt.subplot(2, 2, 2)

    # Check if we have history data
    if all_run_histories and all_run_histories[0]["generations"]:
        # Get generation numbers (assuming they are all the same)
        generations = all_run_histories[0]["generations"]

        # Collect all 'max' fitness histories into a 2D numpy array
        all_max_fitnesses = np.array([run["max_fitness"] for run in all_run_histories])

        # Calculate mean, std dev
        mean_fitness = np.mean(all_max_fitnesses, axis=0)
        std_fitness = np.std(all_max_fitnesses, axis=0)

        # Plot all individual runs (transparent)
        for history in all_run_histories:
            plt.plot(
                history["generations"], history["max_fitness"], color="grey", alpha=0.2
            )

        # Plot the mean line
        plt.plot(
            generations,
            mean_fitness,
            color="blue",
            linewidth=2,
            label="Average Max Fitness",
        )

        # Plot the standard deviation fill
        plt.fill_between(
            generations,
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            color="blue",
            alpha=0.1,
            label="± 1 Std. Dev.",
        )

        plt.title("GA Convergence Plot", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Max Fitness", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
    else:
        plt.text(
            0.5,
            0.5,
            "No convergence data to plot.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("GA Convergence Plot", fontsize=14)

    # --- Plot 3: Histogram of Total Violations ---
    plt.subplot(2, 2, 3)

    if not violation_df.empty:
        total_violations = violation_df.sum(axis=1)

        experiment_runs = np.arange(1, len(total_violations) + 1)

        plt.bar(
            experiment_runs,  # x-values
            total_violations.values,  # y-values
            width=0.6,
            edgecolor="black",
            alpha=0.7,
            color="orange",
        )

        plt.xticks(experiment_runs)

        plt.title(
            f"Total Violations per Experiment Run (N={len(total_violations)})",
            fontsize=14,
        )
        plt.xlabel("Experiment Run", fontsize=12)
        plt.ylabel("Total Violations", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        mean_v = total_violations.mean()
        median_v = total_violations.median()

        plt.axhline(
            mean_v,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_v:.2f}",
        )
        plt.axhline(
            median_v,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_v:.2f}",
        )
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No violation data to plot.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("Total Violations per Experiment Run", fontsize=14)

    # --- Plot 4: Boxplots of Key Analysis Metrics ---
    plt.subplot(2, 2, 4)

    # Select subset of columns from report_df to plot
    cols_to_plot = [
        "avg_age_diff",
        "avg_size_diff",
    ]
    # Filter out columns that might be all NaN (if analysis failed)
    plot_data = report_df[[col for col in cols_to_plot if col in report_df.columns]]

    if not plot_data.empty:
        sns.boxplot(data=plot_data, orient="h")
        plt.title("Distribution of Analysis Metrics (per run)", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Metric", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
    else:
        plt.text(
            0.5,
            0.5,
            "No analysis report data to plot.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("Distribution of Analysis Metrics", fontsize=14)

    plt.tight_layout()
    plt.savefig(figure_filename)
    plt.close()
