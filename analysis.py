#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# MediaPipeの骨格モデルの場所
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "models",
    "pose_landmarker_full.task"
)


# ==========================================
# MediaPipe PoseLandmarker の番号
# ==========================================

# 骨格をつなぐ線
POSE_CONNECTIONS = [

    # 顔
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # 肩
    (11, 12),

    # 左腕
    (11, 13),
    (13, 15),

    # 右腕
    (12, 14),
    (14, 16),

    # 左手
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),

    # 右手
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),

    # 胴体
    (11, 23),
    (12, 24),
    (23, 24),

    # 左脚
    (23, 25),
    (25, 27),

    # 右脚
    (24, 26),
    (26, 28),

    # 左足
    (27, 29),
    (29, 31),
    (27, 31),

    # 右足
    (28, 30),
    (30, 32),
    (28, 32),
]


# ==========================================
# 骨格を動画に描画する
# ==========================================

def draw_skeleton(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    # ランドマークの座標をピクセル座標に変換
    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # ------------------------------
    # 骨格の線
    # ------------------------------

    for start, end in POSE_CONNECTIONS:

        if start >= len(points):
            continue

        if end >= len(points):
            continue

        x1, y1 = points[start]
        x2, y2 = points[end]

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    # ------------------------------
    # 関節の点
    # ------------------------------

    for x, y in points:

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (x, y),
            7,
            (255, 255, 255),
            1
        )


# ==========================================
# 前傾角度を計算
# ==========================================

def calculate_trunk_angle(landmarks):

    try:

        # 左右の肩の中央
        shoulder_x = (
            landmarks[11].x +
            landmarks[12].x
        ) / 2

        shoulder_y = (
            landmarks[11].y +
            landmarks[12].y
        ) / 2

        # 左右の腰の中央
        hip_x = (
            landmarks[23].x +
            landmarks[24].x
        ) / 2

        hip_y = (
            landmarks[23].y +
            landmarks[24].y
        ) / 2

        dx = shoulder_x - hip_x
        dy = hip_y - shoulder_y

        angle = math.degrees(
            math.atan2(
                abs(dx),
                abs(dy)
            )
        )

        return angle

    except Exception:

        return 0.0


# ==========================================
# メイン解析
# ==========================================

def analyze_video(
    video_path,
    output_path,
    runner_height_m=1.65
):

    # モデルが存在するか確認
    if not os.path.exists(MODEL_PATH):

        raise Exception(
            "pose_landmarker_full.task が見つかりません。\n"
            f"現在探している場所:\n{MODEL_PATH}"
        )


    # ----------------------------------
    # 動画を開く
    # ----------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise Exception(
            "動画ファイルを開けませんでした"
        )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0


    # ----------------------------------
    # 出力動画
    # ----------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    if not out.isOpened():

        cap.release()

        raise Exception(
            "出力動画を作成できませんでした"
        )


    # ==================================
    # MediaPipe設定
    # ==================================

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )


    options = vision.PoseLandmarkerOptions(

        base_options=base_options,

        running_mode=
        vision.RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )


    # ==================================
    # 測定用データ
    # ==================================

    frame_count = 0

    detected_frames = 0

    trunk_angles = []

    left_ankle_positions = []

    right_ankle_positions = []


    # ==================================
    # PoseLandmarker開始
    # ==================================

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:


        while True:

            success, frame = cap.read()


            if not success:

                break


            # --------------------------
            # 時間
            # --------------------------

            timestamp_ms = int(
                (
                    frame_count /
                    fps
                ) * 1000
            )


            # --------------------------
            # BGR → RGB
            # --------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # --------------------------
            # MediaPipe画像
            # --------------------------

            mp_image = mp.Image(

                image_format=
                mp.ImageFormat.SRGB,

                data=rgb_frame
            )


            # --------------------------
            # 骨格検出
            # --------------------------

            result = landmarker.detect_for_video(

                mp_image,

                timestamp_ms
            )


            # ==================================
            # 人を検出できた場合
            # ==================================

            if result.pose_landmarks:

                landmarks = (
                    result.pose_landmarks[0]
                )


                detected_frames += 1


                # 骨格を描画
                draw_skeleton(
                    frame,
                    landmarks
                )


                # --------------------------
                # 前傾角度
                # --------------------------

                trunk_angle = (
                    calculate_trunk_angle(
                        landmarks
                    )
                )

                trunk_angles.append(
                    trunk_angle
                )


                # --------------------------
                # 足首位置
                # --------------------------

                left_ankle_positions.append(
                    landmarks[27].x
                )

                right_ankle_positions.append(
                    landmarks[28].x
                )


                # --------------------------
                # 前傾角度を画面表示
                # --------------------------

                cv2.putText(

                    frame,

                    f"Trunk: {trunk_angle:.1f} deg",

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (255, 255, 255),

                    3,

                    cv2.LINE_AA
                )


                cv2.putText(

                    frame,

                    f"Trunk: {trunk_angle:.1f} deg",

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (0, 0, 0),

                    2,

                    cv2.LINE_AA
                )


            # ==================================
            # フレーム保存
            # ==================================

            out.write(
                frame
            )


            frame_count += 1


    # ==================================
    # 終了
    # ==================================

    cap.release()

    out.release()


    # ==================================
    # 平均前傾角度
    # ==================================

    if len(trunk_angles) > 0:

        trunk_lean_deg = float(
            np.mean(
                trunk_angles
            )
        )

    else:

        trunk_lean_deg = 0.0


    # ==================================
    # ストライド推定
    # ==================================

    # 足首の横方向の移動量から簡易推定
    if len(left_ankle_positions) > 1:

        left_range = (
            max(left_ankle_positions)
            -
            min(left_ankle_positions)
        )

        stride_l_m = (
            left_range
            *
            runner_height_m
            *
            1.8
        )

    else:

        stride_l_m = 0.0


    if len(right_ankle_positions) > 1:

        right_range = (
            max(right_ankle_positions)
            -
            min(right_ankle_positions)
        )

        stride_r_m = (
            right_range
            *
            runner_height_m
            *
            1.8
        )

    else:

        stride_r_m = 0.0


    # ==================================
    # ピッチ簡易推定
    # ==================================

    duration_sec = (
        frame_count /
        fps
    )


    if duration_sec > 0 and detected_frames > 0:

        # 仮の歩数推定
        estimated_steps = max(
            1,
            int(
                duration_sec * 3
            )
        )

        cadence_spm = (
            estimated_steps
            /
            duration_sec
            *
            60
        )

    else:

        cadence_spm = 0.0


    # ==================================
    # アドバイス
    # ==================================

    advices = []


    if detected_frames == 0:

        advices.append(
            "人物を検出できませんでした。"
        )

        advices.append(
            "全身が映る横向きの動画を使用してください。"
        )


    else:

        if trunk_lean_deg < 5:

            advices.append(
                "【体幹】上体がやや起きている可能性があります。少し前方へ進む意識を確認しましょう。"
            )

        elif trunk_lean_deg < 15:

            advices.append(
                "【体幹】前傾角度は比較的安定しています。体幹を保ちながら走りましょう。"
            )

        else:

            advices.append(
                "【体幹】前傾が大きめです。腰から折れすぎていないか確認しましょう。"
            )


        if cadence_spm < 150:

            advices.append(
                "【ピッチ】接地時間が長くなっている可能性があります。素早い足さばきを意識しましょう。"
            )

        else:

            advices.append(
                "【ピッチ】テンポを保ちながら走れているか確認しましょう。"
            )


    # ==================================
    # 結果を返す
    # ==================================

    return {

        "frame_count":
        frame_count,

        "detected_frames":
        detected_frames,

        "trunk_lean_deg":
        trunk_lean_deg,

        "cadence_spm":
        cadence_spm,

        "stride_l_m":
        stride_l_m,

        "stride_r_m":
        stride_r_m,

        "advices":
        advices
    }