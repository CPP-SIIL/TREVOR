import requests

OUTSIDE = "outside"
TENTATIVE_IN = "tentative_in"
INSIDE = "inside"
TENTATIVE_OUT = "tentative_out"

class Counter:
    # confirm_threshold originally 40, can also try 120 or other values
    def __init__(self, A, B, confirm_threshold=500):
        self.line_start = A
        self.line_end = B
        self.in_count = 0
        self.out_count = 0

        self.eps = 5
        self.confirm_threshold = confirm_threshold  # pixels from line to confirm crossing

        # track_id -> { "state": ..., "tentative_dir": ... }
        self.id_states = {}

        self.base_url = "http://127.0.0.1:8000"
        self.timeout_seconds = 1.5

    def _ping(self, path):
        url = self.base_url + path
        try:
            r = requests.post(url, timeout=self.timeout_seconds)
            r.raise_for_status()
        except Exception as e:
            print(f"API ping failed ({url}): {e}")

    def _distance_from_line(self, P):
        A = self.line_start
        B = self.line_end
        ABx = B[0] - A[0]
        ABy = B[1] - A[1]
        APx = P[0] - A[0]
        APy = P[1] - A[1]
        cross = abs(ABx * APy - ABy * APx)
        length = (ABx**2 + ABy**2) ** 0.5
        return cross / length if length != 0 else 0

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
            dist = self._distance_from_line(P)

            # First time seeing this ID
            if track_id not in self.id_states:
                if current_side == 1:
                    self.id_states[track_id] = INSIDE
                elif current_side == -1:
                    self.id_states[track_id] = OUTSIDE
                continue  # don't count on first appearance

            state = self.id_states[track_id]

            if state == OUTSIDE:
                if current_side == 1:
                    # Crossed inward — tentatively count
                    self.in_count += 1
                    self._ping("/ingress")
                    self.id_states[track_id] = TENTATIVE_IN

            elif state == TENTATIVE_IN:
                if current_side == -1:
                    # Reversed back out before confirming — undo
                    self.in_count -= 1
                    self._ping("/ingress_undo")
                    self.id_states[track_id] = OUTSIDE
                elif dist >= self.confirm_threshold:
                    # Moved far enough inside — confirmed
                    self.id_states[track_id] = INSIDE

            elif state == INSIDE:
                if current_side == -1:
                    # Crossed outward — tentatively count
                    self.out_count += 1
                    self._ping("/egress")
                    self.id_states[track_id] = TENTATIVE_OUT

            elif state == TENTATIVE_OUT:
                if current_side == 1:
                    # Reversed back in before confirming — undo
                    self.out_count -= 1
                    self._ping("/egress_undo")
                    self.id_states[track_id] = INSIDE
                elif dist >= self.confirm_threshold:
                    # Moved far enough outside — confirmed
                    self.id_states[track_id] = OUTSIDE