from datetime import datetime

from expedantic import ConfigBase


class DisjointConfig(ConfigBase):
    penalty: int  # hard constraint (negative value)


class AgeIntervalConfig(ConfigBase):
    bonus: float
    penalty: float
    target_max_diff: int
    absolute_max_diff: int


class GroupSizeConfig(ConfigBase):
    penalty: float
    hard_penalty: float
    min_size: int
    max_size: int


class Config(ConfigBase):
    num_experiments: int
    current_date: str
    seed: int
    group: str
    run_name: str
    dataset_path: str
    num_groups: int

    num_generations: int
    pop_size: int
    crossover_rate: float
    mutation_rate: float
    elitism_rate: float
    empty_group_penalty: int

    disjoint: DisjointConfig
    age_interval: AgeIntervalConfig
    group_size: GroupSizeConfig

    def model_post_init(self, _context) -> None:
        super().model_post_init(_context)

        today = datetime.now()
        self.current_date = today.strftime("%Y-%m-%d_%H_%M")

        self.seed = int(datetime.now().timestamp())
        self.run_name = f"{self.group}_{self.current_date}_{self.seed}"
