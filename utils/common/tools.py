import numpy as np
import supervision as sv
import cv2
import torch
from torchvision import transforms
from dataclasses import dataclass
from typing import Optional
from utils.common.view import ViewTransformer
from utils.configs.tennis import TennisCourtConfiguration
from utils.classifier.convnet import ConvNeXt


@dataclass
class SnapShotCandidate:
    is_unset: bool = True
    court_ball_xy: Optional[np.ndarray] = None  # position on the bird eye court
    croped_frame: Optional[np.ndarray] = None  # the croped patch of a player
    frames_elapsed: int = 0  # counter before we can confirm this candidate
    confirmed: bool = False
    shot_prediction: Optional[str] = None  # 'backhand', 'forehand', 'serve', 'undefine'
    position: Optional[str] = None  # 'top' or 'bot'
    speed_tracker_xy: Optional[np.ndarray] = None
    speed_tracker_frame_number: Optional[int] = None
    speed_kmh: Optional[int] = None

    def set_snapshot(self, xy, crops, position):
        self.is_unset = False
        self.court_ball_xy = xy
        self.croped_frame = crops
        self.position = position
        self.frames_elapsed = 0

    def reset(self):
        self.is_unset = True
        self.court_ball_xy = None
        self.croped_frame = None
        self.frames_elapsed = 0
        self.confirmed = False
        self.shot_prediction = None
        self.position = None
        self.speed_tracker_xy = None
        self.speed_tracker_frame_number = None
        self.speed_kmh = None

    def increment_frames_elapsed(self):
        self.frames_elapsed += 1

    def confirm_candidate(self, prediction):
        self.confirmed = True
        self.shot_prediction = prediction

    def get_shot_prediction(self):
        assert self.shot_prediction is not None
        return self.shot_prediction

    def get_position(self):
        assert self.position is not None
        return self.position

    def set_speed_tracker(self, position, frame_number, config):
        # we want to pick the section where the speed is mostly easily (accurately) captured
        # we want to avoid the noise from the height of the ball
        # the section just when the ball crosses the net seems like a good option
        if ((self.position == "top") and (position[1] > config.net_y)) or (
            (self.position == "bot") and (position[1] < config.net_y)
        ):
            self.speed_tracker_xy = position
            self.speed_tracker_frame_number = frame_number

    def set_speed(self, end_position, end_frame_number, fps):
        number_of_frames_elapsed = end_frame_number - self.speed_tracker_frame_number
        speed_cms = np.linalg.norm(self.speed_tracker_xy - end_position) / (
            number_of_frames_elapsed / fps
        )  # cm/s
        self.speed_kmh = speed_cms * 3.6 / 100  # km/h


@dataclass
class ResultDisplayer:
    num_frame_remaining: int = 120
    shot_prediction: Optional[str] = None
    speed: Optional[int] = None
    is_set: bool = False

    def display(self):
        assert self.is_set == True
        self.num_frame_remaining -= 1
        if self.num_frame_remaining == 0:
            self.is_set = False
            self.speed = None
        return self.shot_prediction

    def set_display(self, pred, speed):
        self.num_frame_remaining = 120
        self.shot_prediction = pred
        self.speed = speed
        self.is_set = True

    def reset(self):
        self.num_frame_remaining = 120
        self.shot_prediction = None
        self.speed = None
        self.is_set = False


class PatchClassifier:
    def __init__(self, checkpoint, device="cpu"):
        self.model = ConvNeXt()
        self.model.to(device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.img_process = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Normalize(
                    mean=[0.4093, 0.4325, 0.4644], std=[0.1452, 0.1499, 0.1524]
                ),
            ]
        )
        self.ITOS = {0: "backhand", 1: "forehand", 2: "serve", 3: "undefine"}

    def classify(self, np_img, device="cpu"):
        assert np_img.shape == (128, 128, 3)
        np_img = np.ascontiguousarray(
            np_img
        )  # we have negative stride due BGR to RGB conversion, torch don't like that
        patch_ts = (
            torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        )  # (3, 128, 128)
        input = self.img_process(patch_ts)
        input = input.unsqueeze(0)  # (1, 3, 128, 128)
        input = input.to(device)
        with torch.no_grad():
            logit = self.model(input)  # (1, 4)
        values, indices = torch.topk(logit, k=2, dim=1)
        first_pred, second_pred = int(indices[0][0]), int(indices[0][1])
        return (
            self.ITOS[first_pred],
            self.ITOS[second_pred],
        )  # return shot prediction str


def check_during_pt(ball_detections, player_detections, key_points):

    during_pt_frame = True

    if ball_detections.xyxy.shape[0] != 1:
        # no ball or more than 1 ball detected
        during_pt_frame = False

    if player_detections.xyxy.shape[0] != 2:
        # there is no enough player detected or
        # more than two players on court (in case of detecting ball boy)
        during_pt_frame = False

    if (key_points.xy.shape[0] == 0) or (np.sum(key_points.confidence[0] > 0.5) != 14):
        # our expectation for key_points.xy.shape is (1, 14, 2)
        # no tennis court detected
        # or not enough confident key points were detected
        during_pt_frame = False

    return during_pt_frame


def find_center_of_the_court(key_points, config):
    frame_reference_points = key_points.xy[0]
    court_reference_points = np.array(config.vertices)

    transformer = ViewTransformer(  # from bird eye to camera
        source=court_reference_points, target=frame_reference_points
    )

    # finding the middle line
    center_x_bird_eye = (config.singles_x0 + config.singles_x1) / 2
    center_y_bird_eye = config.net_y
    center_of_the_court_bird_eye = np.array([[center_x_bird_eye, center_y_bird_eye]])
    center_of_the_court = transformer.transform_points(
        points=center_of_the_court_bird_eye
    )[
        0
    ]  # (2,)

    return center_of_the_court


def find_scaled_position(point):
    # the scaled position is easier to visualize and has more intuition

    SCALE = 0.5
    PADDING = 200

    scaled_position = (int(point[0] * SCALE) + PADDING, int(point[1] * SCALE) + PADDING)

    return np.array(scaled_position)  # (2,)


def crop_frame(frame, xyxy):
    assert xyxy.shape == (1, 4)
    crop = sv.crop_image(frame, xyxy)
    crop = cv2.resize(crop, (128, 128))  # 128x128 resolution (streched)
    return crop[:, :, ::-1]  # convert to RGB


def find_ball_pos_bird_eye(ball_detection, key_points, config):
    frame_reference_points = key_points.xy[0]
    court_reference_points = np.array(config.vertices)

    transformer = ViewTransformer(  # from camera to bird eye
        source=frame_reference_points, target=court_reference_points
    )

    frame_ball_xy = ball_detection.get_anchors_coordinates(sv.Position.CENTER)  # (1, 2)
    court_ball_xy = transformer.transform_points(points=frame_ball_xy)  # (1, 2)
    return court_ball_xy[0]  # (2,)


def find_player_position(player_detection, key_points, config):
    player_of_interest_xy = player_detection.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )[
        0
    ]  # at player's foot position, (2,)
    center_of_the_court = find_center_of_the_court(key_points, config)  # (2,)
    position = (
        "top" if player_of_interest_xy[1] < center_of_the_court[1] else "bot"
    )  # recall top left corner is (0,0)
    return position


def shot_predictor(
    shot_start: SnapShotCandidate,
    shot_end: SnapShotCandidate,
    config: TennisCourtConfiguration,
):
    left_service_mid = (config.singles_x0 + config.center_x) / 2  # 342.75
    right_service_mid = (config.singles_x1 + config.center_x) / 2  # 754.25

    pred = shot_start.get_shot_prediction()
    pos = shot_start.get_position()

    if pred == "serve":
        if (
            shot_end.court_ball_xy[0] < left_service_mid
            or shot_end.court_ball_xy[0] > right_service_mid
        ):
            # check the x coordinate
            return "serve wide"
        else:
            return "serve t"

    elif pred == "forehand" and pos == "top":
        if (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "forehand cross"
        elif (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "forehand line"
        elif (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "forehand inside out"
        else:
            return "forehand inside in"

    elif pred == "forehand" and pos == "bot":
        if (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "forehand cross"
        elif (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "forehand line"
        elif (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "forehand inside out"
        else:
            return "forehand inside in"

    elif pred == "backhand" and pos == "top":
        if (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "backhand cross"
        elif (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "backhand line"
        elif (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "backhand inside out"
        else:
            return "backhand inside in"

    elif pred == "backhand" and pos == "bot":
        if (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] > config.center_x
        ):
            return "backhand cross"
        elif (shot_start.court_ball_xy[0] < config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "backhand line"
        elif (shot_start.court_ball_xy[0] > config.center_x) and (
            shot_end.court_ball_xy[0] < config.center_x
        ):
            return "backhand inside out"
        else:
            return "backhand inside in"

    else:
        return "no prediction"


def ball_position_crop(position: np.ndarray, config: TennisCourtConfiguration):
    # the vertical height of ball sometime mislead the y coordinate, so we crop it for display
    position = position.copy()
    position[1] = np.clip(position[1], -100, config.length + 100)
    return position
