import requests


class Counter:
    # pointA and pointB represent the start and end of imaginary line
    def __init__(self, A, B):
        self.line_start = A
        self.line_end = B
        self.in_count = 0
        self.out_count = 0
        self.last_side_by_id = {}
        # to work around jitter, treat points very close to line as neutral
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

    # track means object we're tracking
    def update(self, tracks):
        for track_row in tracks:
            # box is in format x1, y1, x2, y2
            box = track_row[0:4]
            track_id = int(track_row[4])
            class_id = int(track_row[6])

            # if this track is not a person, skip and move on to next track
            if class_id != 0:
                continue

            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            P = (center_x, center_y)

            current_side = self.side_of_line(P)

            if current_side == 0:
                continue
            if track_id not in self.last_side_by_id.keys():
                self.last_side_by_id[track_id] = current_side
                continue

            prev_side = self.last_side_by_id[track_id]

            if prev_side == -1 and current_side == 1:
                self.in_count = self.in_count + 1
                self._ping("/ingress")

            elif prev_side == 1 and current_side == -1:
                self.out_count = self.out_count + 1
                self._ping("/egress")

            self.last_side_by_id[track_id] = current_side

    def side_of_line(self, P):
        # A, B, and P are points with x and y coordinates
        # function returns values to represent LEFT, RIGHT, or NEAR
        A = self.line_start
        B = self.line_end

        ABx = B[0] - A[0]
        ABy = B[1] - A[1]
        APx = P[0] - A[0]
        APy = P[1] - A[1]

        value = ABx * APy - ABy * APx

        if abs(value) < self.eps:
            return 0  # 0 means NEAR
        elif value > 0:
            return -1  # -1 means LEFT
        else:
            return 1  # 1 means RIGHT