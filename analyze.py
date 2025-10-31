import platform
from typing import Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from deap import tools
from sklearn.preprocessing import MinMaxScaler

from analysis.analyzer import Analyzer
from analysis.strategies import (
    AgeAnalysisStrategy,
    DisjointAnalysisStrategy,
    GroupSizeAnalysisStrategy,
)
from config import Config


def run_analyzer(
    config: Config,
    people_data: list,
    best_individual: list,
    best_fitness: float,
    logbook: tools.Logbook,
) -> dict:
    strategies = [
        DisjointAnalysisStrategy(),
        AgeAnalysisStrategy(
            config.age_interval.target_max_diff, config.age_interval.absolute_max_diff
        ),
        GroupSizeAnalysisStrategy(
            num_people=len(people_data),
            num_groups=config.num_groups,
            min_size=config.group_size.min_size,
            max_size=config.group_size.max_size,
        ),
    ]

    analyzer = Analyzer(
        people_data=people_data,
        best_individual=best_individual,
        num_groups=config.num_groups,
        strategies=strategies,
    )

    analyzer.save_to_csv(f"./result/report_{config.run_name}.csv")

    print(f"✅ Successfully saved report to 'report_{config.run_name}.csv'.")

    # Get reports and violations
    exp_report = analyzer.get_reports()
    violations = analyzer.get_violations()

    total_violations = sum(violations.values())
    violation_report = {
        "total_violations": total_violations,
        **violations,  # Unpack individual violation counts
    }

    history = {
        "generations": logbook.select("gen"),
        "max_fitness": logbook.select("max"),
    }

    report_df = pd.DataFrame(exp_report)
    report_summary = {
        "avg_age_diff": report_df["Age Diff"].mean(),
        "avg_size_diff": report_df["Size Diff"].mean(),
        "history": history,
        "best_fitness": best_fitness,
        "violation_report": violation_report,
        "exp_report": exp_report,
    }

    columns_to_plot = [
        "Age Diff",
        "Size Diff",
    ]

    # Map the column names to Korean
    label_map = {
        "Age Diff": "나이차",
        "Size Diff": "그룹원 수",
    }

    create_analysis_heatmap(
        exp_report,
        columns_to_plot,
        label_map,
        save_path=f"./result/heatmap_{config.run_name}.png",
        show_plot=False,
    )

    return report_summary


def create_analysis_heatmap(
    reports: list,
    columns_to_plot: list[str],
    label_map: Optional[dict] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Generates a normalized heatmap from the Analyzer's result report.

    Args:
        reports: The reports that analyzed by Analyzer object
        columns_to_plot: List of numeric column names to include in the heatmap
        save_path: (Optional) File path to save the plot (e.g., "heatmap.png")
        show_plot: (Optional) Whether to display the plot (plt.show())
    """

    set_matplotlib_font()  # Set Korean font

    data = pd.DataFrame(reports)

    # Exclude empty groups (Members=0) from the heatmap
    if "Members" in data.columns:
        data = data[data["Members"] > 0]

    # Select only the numeric data for the heatmap
    try:
        # Select only the columns requested by the user
        data_to_plot = data[columns_to_plot].astype(float)
    except KeyError as e:
        print(f"Error: Column '{e}' not found in data.")
        print(f"Available columns: {data.columns.tolist()}")
        return
    except ValueError as e:
        print(f"Error converting data to numeric: {e}")
        return

    # Normalize data (Min-Max Scaling)
    # Scales each column (metric) to a value between 0 and 1 for color consistency.
    scaler = MinMaxScaler()
    data_normalized = pd.DataFrame(
        scaler.fit_transform(data_to_plot),
        columns=data_to_plot.columns,
        index=data_to_plot.index,
    )

    # Adjust figure size based on the number of rows/columns
    fig_height = max(8, len(data_to_plot) * 0.5)
    fig_width = max(12, len(columns_to_plot))
    plt.figure(figsize=(fig_width, fig_height))

    if label_map:
        xtick_labels = [label_map.get(col, col) for col in data_normalized.columns]

    # data: The normalized data that determines the color
    # annot: The text to display in each cell (using the original data)
    # fmt: Format for the annotation (2 decimal places)
    # cmap: Colormap (e.g., 'viridis', 'Blues', 'coolwarm')
    sns.heatmap(
        data_normalized,
        annot=data_to_plot,  # Show original values as text
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={
            "orientation": "horizontal",
            "location": "bottom",
            "shrink": 0.5,  # Colorbar width
            "aspect": 40,  # Colorbar thinkness (bigger -> thinner)
            "pad": 0.03,  # Heatmap and Colorbar
        },
        xticklabels=xtick_labels,
    )

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjusts plot to prevent labels from overlapping

    # Save and/or Show the plot
    if save_path:
        try:
            # bbox_inches='tight' ensures labels aren't cut off
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"✅ Successfully saved heatmap to '{save_path}'.")
        except Exception as e:
            print(f"Error saving plot to '{save_path}': {e}")

    if show_plot:
        plt.show()

    # Close the plot figure to free up memory
    plt.close()


def set_matplotlib_font() -> None:
    """
    Sets the font for Matplotlib/Seaborn to support Korean.
    (Supports Windows, macOS, Linux(Nanum))
    """
    os_name = platform.system()

    if os_name == "Windows":
        font_name = "Malgun Gothic"
    elif os_name == "Darwin":  # macOS
        font_name = "Nanum Gothic"
    elif os_name == "Linux":
        # On Linux, Nanum fonts must be installed.
        # (e.g., sudo apt-get install fonts-nanum*)
        try:
            # Find one of the available Nanum fonts.
            font_path = fm.findfont("NanumGothic", fallback_to_default=False)
            font_name = fm.FontProperties(fname=font_path).get_name()
        except Exception:
            print("Could not find Nanum font. (Try: sudo apt-get install fonts-nanum*)")
            print("Using default font. (Korean may appear broken)")
            font_name = None
    else:
        font_name = None

    if font_name:
        try:
            plt.rc("font", family=font_name)
            print(f"Matplotlib font set to '{font_name}'.")
        except Exception as e:
            print(f"Error setting font: {e}")
            font_name = None

    # If font setting failed or a default font must be used
    if font_name is None:
        # If no font, just print a warning
        print("Could not find a suitable Korean font. Text may appear broken.")

    # Fix for the minus sign (hyphen) breaking
    plt.rc("axes", unicode_minus=False)
