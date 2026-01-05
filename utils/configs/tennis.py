from dataclasses import dataclass, field
from typing import List, Tuple


from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TennisCourtConfiguration:
    """
    Tennis court drawn VERTICALLY:

      x axis: width  (left -> right)
      y axis: length (near baseline at y=0, far baseline at y=length)

    The `vertices` order is the same to the YOLO 14-kpt ground-truth ordering:

      1  Near baseline – left doubles corner
      2  Near baseline – left singles corner
      3  Near baseline – right singles corner
      4  Near baseline – right doubles corner
      5  Far baseline  – right doubles corner
      6  Far baseline  – right singles corner
      7  Far baseline  – left singles corner
      8  Far baseline  – left doubles corner
      9  Near service line – left singles intersection
     10  Far service line  – left singles intersection
     11  Near service line – right singles intersection
     12  Far service line  – right singles intersection
     13  Near service line – center service “T”
     14  Far service line  – center service “T”
    """
    # standard doubles court (cm)
    width: int = 1097     # 10.97 m
    length: int = 2377    # 23.77 m

    # standard singles width (cm)
    singles_width: int = 823  # 8.23 m

    # Service line distance from the net on each side (21 ft = 6.40 m)
    service_line_offset_from_net: int = 640

    @property
    def net_y(self) -> float:
        """center is the net"""
        return self.length / 2

    @property
    def singles_x0(self) -> float:
        """Left singles sideline x (inset from doubles)."""
        return (self.width - self.singles_width) / 2

    @property
    def singles_x1(self) -> float:
        """Right singles sideline x."""
        return (self.width + self.singles_width) / 2

    @property
    def near_service_line_y(self) -> float:
        """Near-side service line y."""
        return self.net_y + self.service_line_offset_from_net

    @property
    def far_service_line_y(self) -> float:
        """Far-side service line y."""
        return self.net_y - self.service_line_offset_from_net

    @property
    def center_x(self) -> float:
        return self.width / 2

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Returns 14 (x, y) vertices in the exact order used by your dataset labels.
        """
        x0, x1 = self.singles_x0, self.singles_x1 # left and right singles line
        y0, y1 = 0.0, float(self.length) # top and bottom baseline
        y_ns, y_fs = float(self.near_service_line_y), float(self.far_service_line_y) # near and far service line
        cx = float(self.center_x) # the center of the court

        return [
            # top left corner is (0,0)
            # 1-4: near baseline (bottom)
            (0.0, y1),             # 1 near-left doubles corner
            (x0, y1),              # 2 near-left singles corner
            (x1, y1),              # 3 near-right singles corner
            (float(self.width), y1),# 4 near-right doubles corner

            # 5-8: far baseline (top)   (note: order is right->left to match your labels)
            (float(self.width), y0),  # 5 far-right doubles corner
            (x1, y0),                 # 6 far-right singles corner
            (x0, y0),                 # 7 far-left singles corner
            (0.0, y0),                # 8 far-left doubles corner

            # 9-12: service line intersections with singles sidelines
            (x0, y_ns),               # 9  near service line – left singles intersection
            (x0, y_fs),               # 10 far  service line – left singles intersection
            (x1, y_ns),               # 11 near service line – right singles intersection
            (x1, y_fs),               # 12 far  service line – right singles intersection

            # 13-14: center service "T" points on service lines
            (cx, y_ns),               # 13 near service line – center service “T”
            (cx, y_fs),               # 14 far  service line – center service “T”
        ]

    # default_factory does is when a new instance of this dataclass is created, 
    # call this lambda function to produce the default value for edges,
    # this avoids the common pitfall of using mutable default arguments (which dataclass don't allow).
    # field() tells a dataclass how an attribute should be created, initialized, compared, and displayed — 
    # instead of just giving it a value.
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # near baseline (doubles) with singles corners on it (horizontal)
        (1, 2), (2, 3), (3, 4),

        # far baseline (doubles) with singles corners on it (horizontal)
        (5, 6), (6, 7), (7, 8),

        # left/right singles sidelines (vertical)
        (2, 7),   # left singles sideline
        (3, 6),   # right singles sideline

        # doubles sidelines (vertical)
        (1, 8),   # left doubles sideline
        (4, 5),   # right doubles sideline

        # service lines (horizontal) 
        (9, 13), (13, 11),   # near service line segment (left -> T -> right)
        (10, 14), (14, 12),  # far service line segment  (left -> T -> right)

        # center service line (between the two T points, vertical)
        (13, 14),

        # connect service intersections up/down the singles sidelines (to form service boxes, vertical)
        (9, 10),
        (11, 12),
    ])
