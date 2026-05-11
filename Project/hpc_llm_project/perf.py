import time
import csv
import os
from collections import defaultdict

class RegionTimer:
    def __init__(self):
        self.times = defaultdict(float)
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self.starts:
            self.times[name] += time.perf_counter() - self.starts[name]
            del self.starts[name]

    def reset(self):
        self.times.clear()
        self.starts.clear()

    def snapshot(self):
        return dict(self.times)

def append_csv(path, row, header=None):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)