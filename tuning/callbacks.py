class MeanAggregator:
    def __init__(self, total_steps: int, last_fraction: float = 0.25):
        self.start_step = int((1.0 - last_fraction) * total_steps)
        self.values = []

    def add(self, step: int, value: float) -> None:
        if step >= self.start_step:
            self.values.append(float(value))

    def objective(self) -> float:
        if not self.values:
            return float("nan")
        return sum(self.values) / len(self.values)
