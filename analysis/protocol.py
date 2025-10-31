from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class AnalysisStrategy(Protocol):
    """
    A protocol for any analysis module.
    It must have an analyze() method that takes a group
    and returns a dictionary of results.
    """

    def analyze(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyzes a single group and returns its metrics.

        Args:
            group: A list of person dictionaries in the group.

        Returns:
            A dictionary of analysis results (e.g., {'Min Age': 20, 'Age Diff': 5}).
        """
        ...
