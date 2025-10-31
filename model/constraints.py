from model.interface import Constraint


class Disjoint(Constraint):
    def __init__(self, penalty: int):
        self.penalty = penalty

    def evaluate(self, group: list) -> float:
        group_ids = {p["id"] for p in group}

        for person in group:
            if not person["disjoint_keys"].isdisjoint(group_ids - {person["id"]}):
                return self.penalty

        return 0.0


class AgeInterval(Constraint):
    def __init__(
        self, bonus: float, penalty: float, target_max_diff: int, absolute_max_diff: int
    ):
        self.bonus = bonus
        self.penalty = penalty
        self.target_max_diff = target_max_diff
        self.absolute_max_diff = absolute_max_diff

    def evaluate(self, group: list) -> float:
        ages = [p["age"] for p in group]

        if not ages or len(ages) < 2:
            return 0.0

        age_diff = max(ages) - min(ages)

        if age_diff > self.absolute_max_diff:
            return -((age_diff - self.absolute_max_diff) * self.penalty)
        elif age_diff > self.target_max_diff:
            return 0.0
        else:
            return self.bonus


class GroupSizeBalance(Constraint):
    """Penalizes deviations from the ideal group size."""

    def __init__(
        self,
        penalty: float,
        hard_penalty: float,
        num_groups: int,
        num_people: int,
        min_size: int,
        max_size: int,
    ):
        self.soft_penalty = penalty
        self.hard_penalty = hard_penalty
        self.num_groups = num_groups
        self.num_people = num_people
        self.min_size = min_size
        self.max_size = max_size

    def evaluate(self, group: list) -> float:
        group_size = len(group)

        violation_amount = 0
        if group_size < self.min_size:
            violation_amount = self.min_size - group_size
        elif group_size > self.max_size:
            violation_amount = group_size - self.max_size

        if violation_amount > 0:
            return -(violation_amount * self.hard_penalty)

        ideal_size = self.num_people / self.num_groups
        size_diff = abs(group_size - ideal_size)
        return -(size_diff * self.soft_penalty)
