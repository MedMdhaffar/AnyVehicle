"""Keep independent ByteTrack counters and age motion in source-frame units."""
from contextlib import contextmanager


class PlateTracker:
    def __init__(self, target):
        from ultralytics.trackers.basetrack import BaseTrack
        self.counter, self.target = BaseTrack, target
        self.value = self.offset = self.maximum = 0
        self.frame_index = None
        self.previous_index = None

    @contextmanager
    def ids(self):
        previous = self.counter._count
        self.counter._count = self.value
        try: yield
        finally:
            self.value = self.counter._count
            self.counter._count = previous

    def update(self, detections, frame):
        from vehicle.common.console_sampling import advance_tracker
        with self.ids():
            if self.frame_index is not None and self.previous_index is not None:
                advance_tracker(self.target, self.frame_index-self.previous_index, detections[:0], frame)
            rows = self.target.update(detections, frame)
        self.previous_index = self.frame_index
        if len(rows):
            rows = rows.copy(); rows[:,4] += self.offset
            self.maximum = max(self.maximum,int(rows[:,4].max()))
        return rows

    def reset(self):
        with self.ids(): self.target.reset()
        self.offset = self.maximum
        self.previous_index = None

    def allocate_id(self):
        """Reserve an ID for a vehicle-owned recovery crop without tracker collisions."""
        with self.ids():
            raw = self.counter.next_id()
        value = raw+self.offset
        self.maximum = max(self.maximum,value)
        return value
