import time


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


class WallClockGuard:
    def __init__(
        self,
        total_steps: int,
        cap_hours: float,
        projection_hours: float,
        projection_min_hours: float = 0.0,
    ):
        self.total_steps = int(total_steps)
        self.cap_hours = cap_hours
        self.projection_hours = projection_hours
        self.projection_min_hours = projection_min_hours
        self.started = time.monotonic()
        self.hit = False
        self.projected_hours = None

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started) / 3600.0

    def exceeded(self, step: int) -> bool:
        elapsed = time.monotonic() - self.started
        if elapsed > self.cap_hours * 3600.0:
            self.hit = True
            return True
        if (
            step > 0
            and elapsed > self.projection_min_hours * 3600.0
            and elapsed * self.total_steps / step > self.projection_hours * 3600.0
        ):
            self.hit = True
            self.projected_hours = elapsed * self.total_steps / step / 3600.0
            return True
        return False

    def reason(self) -> str:
        if self.projected_hours is not None:
            return f"projected {self.projected_hours:.1f}h exceeds wall-clock cap"
        return "wall-clock cap exceeded"
