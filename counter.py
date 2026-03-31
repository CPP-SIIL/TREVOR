import requests


class Counter:
    def __init__(self, A, B):
        self.line_start = A
        self.line_end = B

        self.in_count = 0
        self.out_count = 0

        # Track last valid side (-1 or 1 only)
        self.last_side_by_id = {}

        # Track what directions each ID has already completed
        self.count_history = {}  # track_id -> set(["in", "out"])

        self.eps = 5

        # API settings
        self.base_url = "http://127.0.0.1:8000"
        self.timeout_seconds = 1.5

    def _ping(self, path):
        url = self.base_url + path
        try:
            r = requests.post(url, timeout=self.timeout_seconds)
            r.raise_for_status()
        except Exception as e:
            print(f"API ping failed ({url}): {e}")

    def update(self, tracks):
        for track_row in tracks:
            box = track_row[0:4]
            track_id = int(track_row[4])
            class_id = int(track_row[6])

            if class_id != 0:
                continue

            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            P = (center_x, center_y)

            current_side = self.side_of_line(P)

            # Initialize history
            if track_id not in self.count_history:
                self.count_history[track_id] = set()

            # If already counted both directions → ignore forever
            if len(self.count_history[track_id]) == 2:
                continue

            # First time seeing valid side
            if track_id not in self.last_side_by_id:
                if current_side != 0:
                    self.last_side_by_id[track_id] = current_side
                continue

            prev_side = self.last_side_by_id[track_id]

            # Ignore near-line jitter
            if current_side == 0:
                continue

            # Crossing detection
            if prev_side == -1 and current_side == 1:
                if "in" not in self.count_history[track_id]:
                    self.in_count += 1
                    self.count_history[track_id].add("in")
                    self._ping("/ingress")

            elif prev_side == 1 and current_side == -1:
                if "out" not in self.count_history[track_id]:
                    self.out_count += 1
                    self.count_history[track_id].add("out")
                    self._ping("/egress")

            # Always update side
            self.last_side_by_id[track_id] = current_side

    def side_of_line(self, P):
        A = self.line_start
        B = self.line_end

        ABx = B[0] - A[0]
        ABy = B[1] - A[1]
        APx = P[0] - A[0]
        APy = P[1] - A[1]

        value = ABx * APy - ABy * APx

        if abs(value) < self.eps:
            return 0
        elif value > 0:
            return -1
        else:
            return 1