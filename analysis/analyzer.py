import csv
from collections import Counter
from typing import Any

from analysis.protocol import AnalysisStrategy


class Analyzer:
    """
    A generic analysis engine that runs a list of strategies
    on clustering results and provides reporting methods.
    """

    def __init__(
        self,
        people_data: list[dict[str, Any]],
        best_individual: list[int],
        num_groups: int,
        strategies: list[AnalysisStrategy],
    ):
        """
        Initializes the Analyzer.

        Args:
            people_data: The original list of person data.
            best_individual: A list of group indices assigned to each person.
            num_groups: The total number of groups.
            strategies: A list of objects that conform to the
                        AnalysisStrategy protocol.
        """
        self.people_data = people_data
        self.best_individual = best_individual
        self.num_groups = num_groups
        self.strategies = strategies

        self.group_reports: list[dict[str, Any]] = []
        self.violations: Counter = Counter()
        self._analysis_run: bool = False

    def run_analysis(self) -> None:
        """
        Runs all injected analysis strategies on all groups
        and populates the internal results state.
        """
        if self._analysis_run:
            # Prevent running more than once
            return

        final_groups = [[] for _ in range(self.num_groups)]

        for person_idx, group_idx in enumerate(self.best_individual):
            final_groups[group_idx].append(self.people_data[person_idx])

        for i, group in enumerate(final_groups):
            group_size = len(group)
            group_ids = {p["id"] for p in group}

            final_row_data = {
                "Group": f"Group {i+1}",
                "Members": group_size,
                # "Member IDs": ", ".join(sorted([str(p["id"]) for p in group_ids])),
                "Member IDs": ", ".join(sorted([str(p) for p in group_ids])),
            }

            for strategy in self.strategies:
                try:
                    strategy_results = strategy.analyze(group)

                    # Add the results from this strategy to the final row
                    final_row_data.update(strategy_results)

                    # Automatically detect violations
                    # This assumes a standard: "X Check": "Fail"
                    for key, value in strategy_results.items():
                        if key.endswith(" Check") and value == "Fail":
                            violation_name = key.replace(" Check", "")
                            self.violations[violation_name] += 1

                except Exception as e:
                    strategy_name = strategy.__class__.__name__
                    print(
                        f"WARN: Strategy '{strategy_name}' failed for group {i+1}: {e}"
                    )

            self.group_reports.append(final_row_data)

        self._analysis_run = True

    def get_reports(self) -> list[dict[str, Any]]:
        """
        Returns the raw analysis data.
        Runs the analysis if it hasn't been run yet.
        """
        if not self._analysis_run:
            print("Analysis has not been run. Running now...")
            self.run_analysis()

        return self.group_reports

    def get_violations(self) -> Counter:
        """
        Returns the violation summary.
        Runs the analysis if it hasn't been run yet.
        """
        if not self._analysis_run:
            print("Analysis has not been run. Running now...")
            self.run_analysis()

        return self.violations

    def save_to_csv(self, filename: str) -> None:
        """
        Generically saves the full analysis report to a CSV file.
        It dynamically creates headers from all strategy results.
        """
        if not self._analysis_run:
            self.run_analysis()

        if not self.group_reports:
            print("🪹 No data to save (group_reports is empty).")
            return

        try:
            # Dynamically find all possible headers from all rows
            all_headers = set()

            for row in self.group_reports:
                all_headers.update(row.keys())

            # Create a preferred order for key columns
            ordered_headers = ["Group", "Members", "Member IDs"]
            remaining_headers = sorted(list(all_headers - set(ordered_headers)))
            final_headers = ordered_headers + remaining_headers

            with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                # Use extrasaction='ignore' in case some rows (e.g., empty groups)
                # don't have all the keys.
                writer = csv.DictWriter(
                    f, fieldnames=final_headers, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(self.group_reports)

            print(f"\n💾 Analysis report successfully saved to '{filename}'.")

        except Exception as e:
            print(f"\n❌ An error occurred while saving the CSV file: {e}")
