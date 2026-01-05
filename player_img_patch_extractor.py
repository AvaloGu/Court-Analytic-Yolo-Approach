import os
import cv2
import supervision as sv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.configs.tennis import TennisCourtConfiguration
from utils.common.view import ViewTransformer
from utils.common.tools import (
    SnapShotCandidate,
    find_scaled_position,
    check_during_pt,
    find_player_position,
    find_ball_pos_bird_eye,
    crop_frame,
)
from inference import get_model

ROBOFLOW_API_KEY = "KEY"
PLAYER_DETECTION_MODEL_ID = "tennis-match-cqbju/3"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY
)

# get the trained key point detection model from the Roboflow Universe
FIELD_DETECTION_MODEL_ID = "tennis-court-detection-onesd-ipydn/4"
FIELD_DETECTION_MODEL = get_model(
    model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY
)

BALL_ID = 0
PLAYER_ID = 2

COURT_CONFIG = TennisCourtConfiguration()
EPS_BALL_RADIUS = 175

SNAPSHOT_PATIENCE = 20


def patch_extractor_eps_ball_approach(source_video_path):  

    frame_generator = sv.get_video_frames_generator(source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    during_pt = False
    out_of_pt_frame_counter = 0

    last_hitted_player = ""
    img_counter = 1
    frame_log = []  # for debugging purpose

    # center scaled to the bird eye court
    center_of_the_court_scaled = find_scaled_position(
        (COURT_CONFIG.center_x, COURT_CONFIG.net_y)
    )  # (2,)

    for frame_number, frame in enumerate(
        tqdm(frame_generator, total=video_info.total_frames)
    ):

        # run the model on the frame, specify confidence threshold, why [0] see play.py
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.6)[0]
        detections = sv.Detections.from_inference(result)
        ball_detections = detections[(detections.class_id == BALL_ID)]
        player_detections = detections[detections.class_id == PLAYER_ID]

        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

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
            last_hitted_player = ""
            continue

        if skip_this_frame:
            # not a during pt frame or not enough imformation in this frame, we skip
            continue

        court_reference_points = np.array(COURT_CONFIG.vertices)
        frame_reference_points = key_points.xy[0]

        transformer = ViewTransformer(  # from camera to bird eye
            source=frame_reference_points, target=court_reference_points
        )

        # convert boxes into points, a coordinate on bird eye court
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
        court_ball_xy = transformer.transform_points(points=frame_ball_xy)  # (1, 2)

        players_xy = player_detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        court_players_xy = transformer.transform_points(points=players_xy)  # (2, 2)

        ball_position_scaled = find_scaled_position(court_ball_xy[0])  # (2,)

        for id, player in enumerate(court_players_xy):
            scaled_position = find_scaled_position(player)  # (2,)
            if (
                scaled_position[1] < center_of_the_court_scaled[1]
            ):  # recall top left corner is (0,0)
                current_player = "top"
            else:
                current_player = "bot"

            if last_hitted_player == current_player:
                # this player just hitted, it will be the other player's turn
                continue
            elif (
                np.linalg.norm(scaled_position - ball_position_scaled) > EPS_BALL_RADIUS
            ) and (last_hitted_player != ""):
                # not within the epsilon ball.
                # At the start of a point, some player has really high toss that might exceed the epsilon ball
                continue
            else:
                # the player whom is about to hit and the ball is currently
                # within the epsilon ball region (i.e. its close enough),
                crops = sv.crop_image(frame, player_detections[id].xyxy)
                crops = cv2.resize(crops, (128, 128))  # 128x128 resolution (stretched)
                crops = crops[:, :, ::-1]  # we will save it as RGB
                plt.imsave(f"img_folder/img_{img_counter}.png", crops)

                img_counter += 1
                last_hitted_player = current_player  # update the last hitted player once the player hitted
                frame_log.append(frame_number)

    # np.save(frame_log.npy, frame_log)


def patch_extractor_vertical_dist_approach(source_video_path):

    frame_generator = sv.get_video_frames_generator(source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    during_pt = False
    out_of_pt_frame_counter = 0

    shot_start = SnapShotCandidate()
    shot_end = SnapShotCandidate()

    img_counter = 1
    frame_log = []

    for frame_number, frame in enumerate(
        tqdm(frame_generator, total=video_info.total_frames)
    ):

        # run the model on the frame, specify confidence threshold, why [0] see play.py
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.6)[0]
        detections = sv.Detections.from_inference(result)
        ball_detections = detections[(detections.class_id == BALL_ID)]
        player_detections = detections[detections.class_id == PLAYER_ID]

        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

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
            shot_start.reset()
            shot_end.reset()
            continue

        if skip_this_frame:
            # not a during pt frame or not enough imformation in this frame, we skip
            if not shot_end.is_unset:  # we'll still increment the counter
                shot_end.increment_frames_elapsed()
            continue

        players_xy = player_detections.get_anchors_coordinates(
            sv.Position.CENTER
        )  # (2, 2)
        ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]  # (2,)

        # fiding the player that is about to hit a shot
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
                continue

        # ball position in bird eye court
        ball_position = find_ball_pos_bird_eye(
            ball_detections, key_points, COURT_CONFIG
        )  # (2,)
        # path.append(ball_position_crop(ball_position, CONFIG))

        if shot_start.is_unset:
            # start of a point, most of the time when all required information
            # were detected is at the moment of serve preparation or tossing motion
            patch = crop_frame(frame, player_detection_of_interest.xyxy)
            shot_start.set_snapshot(ball_position, patch, position)
            shot_start.confirm_candidate(prediction="serve")
            plt.imsave(f"img_folder/img_{img_counter}.png", patch)
            img_counter += 1
            frame_log.append(frame_number)

        elif shot_end.is_unset:
            assert position != shot_start.get_position()
            patch = crop_frame(frame, player_detection_of_interest.xyxy)
            shot_end.set_snapshot(ball_position, patch, position)

        elif shot_end.frames_elapsed > SNAPSHOT_PATIENCE:
            # we confirm this snapshot is the point of contact
            shot_end.confirm_candidate("undefine")
            plt.imsave(f"img_folder/img_{img_counter}.png", shot_end.croped_frame)
            img_counter += 1
            frame_log.append(frame_number)
            shot_start = shot_end
            shot_end = SnapShotCandidate()

        else:  # shot_end is set but not yet confirmed
            if shot_end.get_position() == "top":  # top left (0,0)
                if ball_position[1] < shot_end.court_ball_xy[1]:  # compare y coordinate
                    # update the candidate
                    patch = crop_frame(frame, player_detection_of_interest.xyxy)
                    shot_end.set_snapshot(ball_position, patch, position)
                else:  # the current candidate is still the most likely to be the point of contacct
                    shot_end.increment_frames_elapsed()
            else:  # shot_end.get_position() == 'bot', top left (0,0)
                if ball_position[1] > shot_end.court_ball_xy[1]:  # compare y coordinate
                    # update the candidate
                    patch = crop_frame(frame, player_detection_of_interest.xyxy)
                    shot_end.set_snapshot(ball_position, patch, position)
                else:
                    shot_end.increment_frames_elapsed()

    # np.save(frame_log.npy, frame_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch extractor")
    parser.add_argument(
        "extractor", help="which extractor to run, eps_ball or vertical_dist"
    )
    parser.add_argument("video_path", help="source video path")
    args = parser.parse_args()

    directory_path = "img_folder"
    os.makedirs(directory_path, exist_ok=True)

    if args.extractor == "eps_ball":
        patch_extractor_eps_ball_approach(args.video_path)
    else:
        patch_extractor_vertical_dist_approach(args.video_path)
