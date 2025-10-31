from abc import ABC, abstractmethod


class Constraint(ABC):
    @abstractmethod
    def evaluate(self, group: list) -> float:
        """
        Evaluates the constraint for a given group and returns a fitness score.

        Args:
            group: The group to evaluate.

        Returns:
            The fitness score for the constraint.
        """
        pass
