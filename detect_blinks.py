# 导入工具包
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import numpy as np
import argparse
import time
import dlib
import cv2
import winsound  # 声音报警（Windows）
import threading  # 多线程报警
from imutils.video import FileVideoStream  # 多线程视频流

# -------------------------- 1. 常量定义 --------------------------
# 68点人脸关键点索引
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# 颜色定义
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)

# -------------------------- 2. 核心计算函数 --------------------------
def eye_aspect_ratio(eye):
    """计算眼睛纵横比（EAR）"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """计算嘴巴纵横比（MAR），用于检测打哈欠"""
    # 竖直距离
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    # 水平距离
    D = dist.euclidean(mouth[12], mouth[16])
    # MAR值
    mar = (A + B + C) / (3.0 * D)
    return mar

def get_head_pose(shape, frame_size):
    """基于关键点计算头部姿态（欧拉角：俯仰角pitch、偏航角yaw、滚转角roll）"""
    # 2D关键点（选取鼻尖、双眼中心、下巴、左右嘴角）
    image_points = np.array([
        shape[30],    # 鼻尖
        shape[8],     # 下巴
        shape[36],    # 左眼左角
        shape[45],    # 右眼右角
        shape[48],    # 左嘴角
        shape[54]     # 右嘴角
    ], dtype=np.float32)

    # 3D模型点（标准人脸模型）
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 鼻尖
        (0.0, -330.0, -65.0),        # 下巴
        (-225.0, 170.0, -135.0),     # 左眼左角
        (225.0, 170.0, -135.0),      # 右眼右角
        (-150.0, -150.0, -125.0),    # 左嘴角
        (150.0, -150.0, -125.0)      # 右嘴角
    ])

    # 相机内参
    (h, w) = frame_size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )

    # 畸变系数（假设无畸变）
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 求解PnP问题
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # 转换旋转向量为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 计算欧拉角
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]
    return pitch, yaw, roll

# -------------------------- 3. 报警函数 --------------------------
def sound_alarm():
    """声音报警（独立线程，避免阻塞）"""
    for _ in range(3):
        winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
        time.sleep(0.1)

def visual_alarm(frame):
    """视觉报警：红色边框+闪烁文字"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), COLOR_RED, 5)
    cv2.putText(frame, "FATIGUE ALERT!!!", (w//4, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_RED, 3)
    return frame

# -------------------------- 4. 主函数 --------------------------
def main():
    # 解析命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
                    help="path to input video file (empty for camera)")
    args = vars(ap.parse_args())

    # -------------------------- 初始化参数 --------------------------
    # 基础阈值
    EYE_AR_THRESH = 0.3  # 闭眼EAR阈值
    EYE_AR_CONSEC_FRAMES = 3  # 有效眨眼连续帧数
    MAR_THRESH = 0.6  # 打哈欠MAR阈值
    PITCH_THRESH = 20  # 低头俯仰角阈值（度）
    YAW_THRESH = 30  # 歪头偏航角阈值（度）

    # 时间统计参数
    BLINK_WINDOW = 60  # 统计眨眼频率的时间窗口（秒）
    LONG_CLOSE_THRESH = 1.0  # 长时间闭眼阈值（秒）
    YAWN_CONSEC_FRAMES = 5  # 有效打哈欠连续帧数

    # 计数器初始化
    COUNTER = 0  # 连续闭眼帧数
    TOTAL_BLINKS = 0  # 总眨眼次数
    YAWN_COUNTER = 0  # 连续打哈欠帧数
    TOTAL_YAWNS = 0  # 总打哈欠次数
    LONG_CLOSE_COUNTER = 0  # 长时间闭眼帧数
    MAX_CLOSE_DURATION = 0  # 最长闭眼时长（秒）

    # 滑动窗口存储眨眼时间戳（用于计算频率）
    blink_timestamps = deque(maxlen=int(BLINK_WINDOW * 30))  # 假设30fps
    frame_counter = 0  # 总帧数
    fps = 30  # 初始帧率（后续动态计算）
    alarm_triggered = False  # 报警标志

    # 疲劳等级评分项（每项0-10分，总分0-40）
    score_blink = 0  # 眨眼频率评分
    score_yawn = 0  # 打哈欠评分
    score_long_close = 0  # 长时间闭眼评分
    score_head_pose = 0  # 头部姿态评分

    # -------------------------- 加载模型和视频流 --------------------------
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # 提取关键点索引范围
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (mStart, mEnd) = FACIAL_LANDMARKS_68_IDXS["mouth"]

    # 启动视频流（多线程加速）
    print("[INFO] starting video stream...")
    if args["video"]:
        vs = FileVideoStream(args["video"]).start()
    else:
        vs = FileVideoStream(0).start()  # 摄像头
    time.sleep(1.0)

    # 记录开始时间
    start_time = time.time()

    # -------------------------- 帧处理循环 --------------------------
    while True:
        # 读取帧
        frame = vs.read()
        if frame is None:
            break

        # 预处理：调整尺寸、转灰度
        (h, w) = frame.shape[:2]
        resize_w = 800  # 缩小尺寸提高速度
        r = resize_w / float(w)
        dim = (resize_w, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[:2]

        # 动态计算帧率
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        # -------------------------- 人脸检测与关键点提取 --------------------------
        rects = detector(gray, 0)
        if len(rects) == 0:
            # 未检测到人脸
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        else:
            for rect in rects:
                # 预测关键点并转换为数组
                shape = predictor(gray, rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=int)

                # 提取各部位关键点
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                # -------------------------- 特征计算 --------------------------
                # 计算EAR和MAR
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # 计算头部姿态
                pitch, yaw, roll = get_head_pose(shape, frame_size)

                # -------------------------- 眨眼检测 --------------------------
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    LONG_CLOSE_COUNTER += 1
                    # 计算当前闭眼时长
                    close_duration = LONG_CLOSE_COUNTER / fps
                    MAX_CLOSE_DURATION = max(MAX_CLOSE_DURATION, close_duration)
                else:
                    # 统计有效眨眼
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINKS += 1
                        blink_timestamps.append(time.time())
                    # 重置计数器
                    COUNTER = 0
                    LONG_CLOSE_COUNTER = 0

                # -------------------------- 打哈欠检测 --------------------------
                if mar > MAR_THRESH:
                    YAWN_COUNTER += 1
                    if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                        TOTAL_YAWNS += 1
                        YAWN_COUNTER = 0
                else:
                    YAWN_COUNTER = 0

                # -------------------------- 特征统计 --------------------------
                # 计算眨眼频率（次/分钟）
                current_time = time.time()
                while blink_timestamps and current_time - blink_timestamps[0] > BLINK_WINDOW:
                    blink_timestamps.popleft()
                blink_frequency = len(blink_timestamps) * 60 / BLINK_WINDOW if BLINK_WINDOW > 0 else 0

                # -------------------------- 疲劳等级评分 --------------------------
                # 1. 眨眼频率评分（正常：10-20次/分钟，超出则扣分）
                if 10 <= blink_frequency <= 20:
                    score_blink = 0
                else:
                    score_blink = min(10, abs(blink_frequency - 15) // 2)

                # 2. 打哈欠评分（每打一次加2分，最多10分）
                score_yawn = min(10, TOTAL_YAWNS // 1)

                # 3. 长时间闭眼评分（超过1秒加5分，最多10分）
                score_long_close = min(10, int(MAX_CLOSE_DURATION) * 5)

                # 4. 头部姿态评分（低头/歪头加5分，最多10分）
                score_head_pose = 0
                if abs(pitch) > PITCH_THRESH or abs(yaw) > YAW_THRESH:
                    score_head_pose = 10

                # 总评分（0-40）
                total_score = score_blink + score_yawn + score_long_close + score_head_pose

                # 疲劳等级判定
                if total_score < 5:
                    fatigue_level = "NORMAL"
                    fatigue_color = COLOR_GREEN
                    alarm_triggered = False
                elif total_score < 15:
                    fatigue_level = "MILD TIRED"
                    fatigue_color = COLOR_YELLOW
                    alarm_triggered = False
                elif total_score < 25:
                    fatigue_level = "MODERATE TIRED"
                    fatigue_color = COLOR_YELLOW
                    alarm_triggered = False
                else:
                    fatigue_level = "SEVERE FATIGUE"
                    fatigue_color = COLOR_RED
                    # 触发报警
                    if not alarm_triggered:
                        alarm_triggered = True
                        threading.Thread(target=sound_alarm, daemon=True).start()
                    frame = visual_alarm(frame)

                # -------------------------- 绘制可视化 --------------------------
                # 绘制眼部和嘴巴轮廓
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, COLOR_GREEN, 1)
                cv2.drawContours(frame, [rightEyeHull], -1, COLOR_GREEN, 1)
                cv2.drawContours(frame, [mouthHull], -1, COLOR_BLUE, 1)

                # 绘制文本信息
                # 基础统计
                cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                cv2.putText(frame, f"Yawns: {TOTAL_YAWNS}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2)

                # 头部姿态
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (350, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (350, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2)

                # 疲劳相关
                cv2.putText(frame, f"Blink Freq: {blink_frequency:.1f}/min", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                cv2.putText(frame, f"Max Close: {MAX_CLOSE_DURATION:.2f}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                cv2.putText(frame, f"Fatigue: {fatigue_level}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
                cv2.putText(frame, f"Score: {total_score}/40", (250, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, fatigue_color, 2)

        # 显示帧率
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)

        # 显示帧
        cv2.imshow("Fatigue Detection System", frame)
        key = cv2.waitKey(1) & 0xFF

        # 按ESC退出
        if key == 27:
            break

    # -------------------------- 资源释放 --------------------------
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()