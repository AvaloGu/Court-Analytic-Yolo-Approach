import os
import argparse
import numpy as np
import torch
import cv2
import supervision as sv
import pdb  # for debug
from tqdm import tqdm
from inference import get_model
from utils.configs.tennis import TennisCourtConfiguration
from utils.annotators.tennis import FrameAnnotator, draw_tennis_court
from utils.common.tools import (
    PatchClassifier,
    SnapShotCandidate,
    ResultDisplayer,
    check_during_pt,
    find_player_position,
    shot_predictor,
    find_ball_pos_bird_eye,
    crop_frame,
)

# !pip install -q inference-gpu supervision
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

ROBOFLOW_API_KEY = "KEY"
PLAYER_DETECTION_MODEL_ID = "tennis-match-cqbju/3"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY
)
FIELD_DETECTION_MODEL_ID = "tennis-court-detection-onesd-ipydn/4"
FIELD_DETECTION_MODEL = get_model(
    model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY
)

BALL_ID = 0
PLAYER_ID = 2

COURT_CONFIG = TennisCourtConfiguration()
SNAPSHOT_PATIENCE = 20 # number of frames of patience


def inference(video_folder):

    source_video_path = video_folder
    target_video_path = video_folder.split(".")[0] + "_result.mov"
    target_video_path_be = video_folder.split(".")[0] + "_result_bird_eye.mov"

    frame_generator = sv.get_video_frames_generator(source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    frame_annotator = FrameAnnotator()

    # Build first court image to determine output resolution
    court0 = draw_tennis_court(COURT_CONFIG)

    # video out_info
    out_info_bird_eye = sv.VideoInfo(
        width=court0.shape[1],
        height=court0.shape[0],
        fps=video_info.fps,
        total_frames=video_info.total_frames,
    )

    video_sink = sv.VideoSink(target_video_path, video_info=video_info)
    video_sink_bird_eye = sv.VideoSink(
        target_video_path_be, video_info=out_info_bird_eye
    )

    during_pt = False
    out_of_pt_frame_counter = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("model.pth", map_location=device)
    patch_classifier = PatchClassifier(checkpoint, device)

    shot_start = SnapShotCandidate()
    shot_end = SnapShotCandidate()

    result_displayer = ResultDisplayer()

    path = []

    with video_sink, video_sink_bird_eye:
        for frame_number, frame in enumerate(
            tqdm(frame_generator, total=video_info.total_frames)
        ):

            # run the model on the frame, specify confidence threshold, why [0] see play.py
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.6)[0]
            detections = sv.Detections.from_inference(result)
            annotated_frame = frame_annotator.annotate_video_frame(frame, detections)
            if result_displayer.is_set:
                pred = result_displayer.display()
                cv2.putText(
                    annotated_frame,
                    f"Shot type: {pred}",
                    (600, 900),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if result_displayer.speed is not None:
                    cv2.putText(
                        annotated_frame,
                        f"Shot speed: {result_displayer.speed} km/h",
                        (600, 950),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            video_sink.write_frame(annotated_frame)

            result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            key_points = sv.KeyPoints.from_inference(result)

            ball_detections = detections[(detections.class_id == BALL_ID)]
            player_detections = detections[detections.class_id == PLAYER_ID]
            court = frame_annotator.annotate_court_frame(
                ball_detections, player_detections, key_points, path, COURT_CONFIG
            )

            # if frame_number == 2107 or frame_number == 2141: # conditional breakpoint
            #     pdb.set_trace()

            if check_during_pt(ball_detections, player_detections, key_points):
                during_pt = True
                out_of_pt_frame_counter = 0
                skip_this_frame = False
            else:
                out_of_pt_frame_counter += 1
                skip_this_frame = True

            if out_of_pt_frame_counter > video_info.fps * 2:
                # consistently out of point for around 2 seconds.
                during_pt = False

            if during_pt is False:
                # we reset if we got out of the point, and skip the frame
                if not shot_end.is_unset:
                    first_pred, second_pred = patch_classifier.classify(
                        shot_end.croped_frame, device
                    )
                    pred = (
                        second_pred
                        if first_pred == "undefine" or first_pred == "serve"
                        else first_pred
                    )
                    shot_end.confirm_candidate(pred)
                    shot_pred = shot_predictor(shot_start, shot_end, COURT_CONFIG)
                    result_displayer.set_display(shot_pred)

                shot_start.reset()
                shot_end.reset()
                video_sink_bird_eye.write_frame(court)
                path = []
                continue

            if skip_this_frame:
                # not a during pt frame or not enough imformation in this frame, we skip
                if not shot_end.is_unset:  # we'll still increment the counter
                    shot_end.increment_frames_elapsed()
                video_sink_bird_eye.write_frame(court)
                continue

            players_xy = player_detections.get_anchors_coordinates(
                sv.Position.CENTER
            )  # (2, 2)
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[
                0
            ]  # (2,)

            # finding the player that is about to hit a shot
            hitting_player_idx = int(
                np.argmin(np.linalg.norm(ball_xy - players_xy, axis=1))
            )
            player_detection_of_interest = player_detections[hitting_player_idx]
            position = find_player_position(
                player_detection_of_interest, key_points, COURT_CONFIG
            )  # top or bot

            if (not shot_start.is_unset) and (position == shot_start.position):
                # we want to make sure when the shot_end is set, it must be on the opposite side of the court
                if shot_end.is_unset:
                    # the ball is still traveling away from the player who last hitted
                    # this frame is not useful to us
                    video_sink_bird_eye.write_frame(court)
                    continue

            # ball position in bird eye court
            ball_position = find_ball_pos_bird_eye(
                ball_detections, key_points, COURT_CONFIG
            )  # (2,)
            # path.append(ball_position_crop(ball_position, CONFIG))

            if not shot_start.is_unset:
                if shot_start.speed_tracker_xy is None:
                    shot_start.set_speed_tracker(
                        ball_position, frame_number, COURT_CONFIG
                    )
                else:  # the starting position of the speed tracker is already set
                    if shot_start.speed_kmh is None:
                        shot_start.set_speed(
                            ball_position, frame_number, video_info.fps
                        )

            if shot_start.is_unset:
                # start of a point
                patch = crop_frame(frame, player_detection_of_interest.xyxy)
                first_pred, second_pred = patch_classifier.classify(patch, device)
                if first_pred == "serve":
                    shot_start.set_snapshot(ball_position, patch, position)
                    shot_start.confirm_candidate(prediction=first_pred)
                else:
                    video_sink_bird_eye.write_frame(court)
                    continue

            elif shot_end.is_unset:
                assert position != shot_start.get_position()
                patch = crop_frame(frame, player_detection_of_interest.xyxy)
                shot_end.set_snapshot(ball_position, patch, position)

            elif shot_end.frames_elapsed > SNAPSHOT_PATIENCE:
                # we confirm this snapshot is the point of contact
                first_pred, second_pred = patch_classifier.classify(
                    shot_end.croped_frame, device
                )
                pred = (
                    second_pred
                    if first_pred == "undefine" or first_pred == "serve"
                    else first_pred
                )
                shot_end.confirm_candidate(pred)
                shot_pred = shot_predictor(shot_start, shot_end, COURT_CONFIG)
                shot_speed = int(shot_start.speed_kmh)
                shot_start = shot_end
                shot_end = SnapShotCandidate()
                result_displayer.set_display(shot_pred, shot_speed)

            else:  # shot_end is set but not yet confirmed
                if shot_end.get_position() == "top":  # top left (0,0)
                    if (
                        ball_position[1] < shot_end.court_ball_xy[1]
                    ):  # compare y coordinate
                        # update the candidate
                        patch = crop_frame(frame, player_detection_of_interest.xyxy)
                        shot_end.set_snapshot(ball_position, patch, position)
                    else:  # the current candidate is still the most likely to be the point of contacct
                        shot_end.increment_frames_elapsed()
                else:  # shot_end.get_position() == 'bot', top left (0,0)
                    if (
                        ball_position[1] > shot_end.court_ball_xy[1]
                    ):  # compare y coordinate
                        # update the candidate
                        patch = crop_frame(frame, player_detection_of_interest.xyxy)
                        shot_end.set_snapshot(ball_position, patch, position)
                    else:
                        shot_end.increment_frames_elapsed()

            video_sink_bird_eye.write_frame(court)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tennis ai")
    parser.add_argument("video_path", help="video source file")
    args = parser.parse_args()
    inference(args.video_path)
