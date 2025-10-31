from typing import Any


class DisjointAnalysisStrategy:
    """Analyzes the 'disjoint' constraint."""

    def __init__(self):
        pass

    def analyze(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {}

        group_ids = {p["id"] for p in group}

        violation = any(
            not person["disjoint_keys"].isdisjoint(group_ids - {person["id"]})
            for person in group
        )

        return {"Disjoint Check": "Fail" if violation else "Pass"}


class AgeAnalysisStrategy:
    """Analyzes the age distribution within a group."""

    def __init__(self, target_max_diff: int, absolute_max_diff: int):
        self.target_max_diff = target_max_diff
        self.absolute_max_diff = absolute_max_diff

    def analyze(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {}

        ages = [p["age"] for p in group]
        min_age, max_age = min(ages), max(ages)
        age_diff = max_age - min_age

        check_pass = age_diff <= self.absolute_max_diff

        return {
            "Min Age": min_age,
            "Max Age": max_age,
            "Age Diff": age_diff,
            "Age Check": "Pass" if check_pass else "Fail",
        }


class GroupSizeAnalysisStrategy:
    """Analyzes the deviation from the ideal group size."""

    def __init__(self, num_people: int, num_groups: int, min_size: int, max_size: int):
        self.ideal_size = 0.0
        self.min_size = min_size
        self.max_size = max_size

        if num_groups > 0:
            self.ideal_size = num_people / num_groups

    def analyze(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        if self.ideal_size == 0.0:
            return {"Size Check": "N/A"}

        group_size = len(group)
        size_diff = abs(group_size - self.ideal_size)

        check_pass = self.min_size <= group_size <= self.max_size

        return {
            "Group Size": group_size,
            "Ideal Size": round(self.ideal_size, 2),
            "Size Diff": round(size_diff, 2),
            "Size Check": "Pass" if check_pass else "Fail",
        }
