from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from utils.configs.tennis import TennisCourtConfiguration
from utils.common.view import ViewTransformer

import numpy as np
import cv2
import supervision as sv

def draw_tennis_court(
    config,
    outer_color: sv.Color = sv.Color(90, 130, 115),   # green surround 
    court_color: sv.Color = sv.Color(95, 110, 150),   # bluish court fill
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 200,
    line_thickness: int = 1,
    point_radius: int = 5,
    scale: float = 0.5,
    draw_points: bool = True,
) -> np.ndarray:
    """
    Draw a tennis court (vertical) using config.vertices (cm) + config.edges (1-based indices).
    Produces a hard-court style image: green surround + blue court + white lines.
    """

    scaled_w = int(config.width * scale)
    scaled_l = int(config.length * scale)

    # background (greenish surround)
    img = np.ones((scaled_l + 2 * padding, scaled_w + 2 * padding, 3), dtype=np.uint8)
    img[:] = np.array(outer_color.as_bgr(), dtype=np.uint8)

    def to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
        # image coordinates are (col=x, row=y)
        # add padding since we want a 'image origin' shift for better display (move origin away from top-left)
        x = int(x_cm * scale) + padding
        y = int(y_cm * scale) + padding
        return (x, y)

    # fill the court surface in bluish us open color
    court_poly = np.array([
        to_px(0.0, 0.0),
        to_px(config.width, 0.0),
        to_px(config.width, config.length),
        to_px(0.0, config.length),
    ], dtype=np.int32)
    cv2.fillPoly(img, [court_poly], color=court_color.as_bgr())

    # draw the white court lines from edges
    verts = config.vertices  # list of (x_cm, y_cm) in the vertices ordering
    for start, end in config.edges: # edges are 1-based indices
        x1, y1 = verts[start - 1]
        x2, y2 = verts[end - 1]
        cv2.line(
            img,
            to_px(x1, y1),
            to_px(x2, y2),
            color=line_color.as_bgr(),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    # net line (across doubles width)
    yN = config.net_y
    cv2.line(
        img,
        to_px(0.0, yN),
        to_px(config.width, yN),
        color=line_color.as_bgr(),
        thickness=max(1, line_thickness - 1),
        lineType=cv2.LINE_AA,
    )

    # draw keypoints
    # if draw_points:
    #     for (x_cm, y_cm) in verts:
    #         cv2.circle(img, to_px(x_cm, y_cm), point_radius, (255, 105, 180), -1, lineType=cv2.LINE_AA)

    return img


def draw_points_on_court(
    config: TennisCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 200,
    scale: float = 0.5,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a tennic court diagram.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if court is None:
        court = draw_tennis_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1 # the circle is drawn as a solid, filled disk
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness # draws only the circumference or edge of the circle
        )

    return court


def draw_path_on_court(
    config: TennisCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 200,
    scale: float = 0.5,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
   
    if court is None:
        court = draw_tennis_court(
            config=config,
            padding=padding,
            scale=scale
        )


    scaled_path = [
        (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        for point in paths
    ]

    if len(scaled_path) < 2:
        return court

    for i in range(len(scaled_path) - 1):
        cv2.line(
            img=court,
            pt1=scaled_path[i],
            pt2=scaled_path[i + 1],
            color=color.as_bgr(),
            thickness=thickness
        )

    return court


class FrameAnnotator():
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF']),
            thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF']),
            text_color=sv.Color.from_hex('#000000'))
        
    def annotate_video_frame(self, frame, detection):
        labels = [ # labels we want to show in the annotation
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detection['class_name'], detection.confidence)
        ]
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate( # bounding boxes
            scene=annotated_frame,
            detections=detection)
        annotated_frame = self.label_annotator.annotate( # label annotation
            scene=annotated_frame,
            detections=detection,
            labels=labels)
        return annotated_frame
    
    def annotate_court_frame(self, ball_detections, player_detections, key_points, path, config):
        court = draw_tennis_court(config)
        if (key_points.xy.shape[0] == 0) or (np.sum(key_points.confidence[0] > 0.5) < 10):
            return court
        
        # recall key_points.confidence is (N, K), (N, 14)
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]

        court_reference_points = np.array(config.vertices)[filter]

        transformer = ViewTransformer( # from camera to bird eye
            source=frame_reference_points,
            target=court_reference_points
        )

        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
        court_ball_xy = transformer.transform_points(points=frame_ball_xy)

        players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_players_xy = transformer.transform_points(points=players_xy)

        court = draw_points_on_court(config=config, 
                                    xy=court_ball_xy, 
                                    court=court)
        court = draw_points_on_court(config=config, 
                                    xy=court_players_xy, 
                                    face_color=sv.Color.from_hex('00BFFF'),
                                    edge_color=sv.Color.BLACK,
                                    radius=16,
                                    court=court)
        court = draw_path_on_court(config=config, 
                                   paths=path, 
                                   color=sv.Color.WHITE,
                                   court=court)
        return court