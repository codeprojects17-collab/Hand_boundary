# hand_boundary_poc.py
import cv2
import numpy as np
import time
import argparse
import math

parser = argparse.ArgumentParser(description="Hand-boundary interaction POC")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--scale", type=float, default=0.6, help="scale down frame for speed (0.3-1.0)")
parser.add_argument("--draw-rect", action="store_true", help="draw rectangle boundary (default: circle/box)")
parser.add_argument("--debug-mask", action="store_true", help="show skin mask window")
parser.add_argument("--min-area", type=int, default=2000, help="min contour area to consider a hand")
parser.add_argument("--safe-ratio", type=float, default=0.25, help="safe threshold ratio of frame diagonal")
parser.add_argument("--warning-ratio", type=float, default=0.07, help="warning threshold ratio of frame diagonal")
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Helper: skin mask using YCrCb + HSV heuristics
def skin_mask(frame_bgr):
    # Blur to reduce noise
    f = cv2.GaussianBlur(frame_bgr, (5,5), 0)
    # YCrCb
    ycrcb = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) = cv2.split(ycrcb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0,133,77]), np.array([255,173,127]))
    # HSV
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([25, 150, 255]))
    # Combined mask
    mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# Distance from point to rectangle boundary (0 if inside)
def point_to_rect_distance(px, py, x1, y1, x2, y2):
    dx = 0
    dy = 0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    return math.hypot(dx, dy)

prev_time = time.time()
fps_smooth = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Scale frame for speed
    h0, w0 = frame.shape[:2]
    scale = max(0.3, min(1.0, args.scale))
    frame = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
    h, w = frame.shape[:2]

    # Virtual boundary: centered rectangle (you can change size)
    rect_w = int(w * 0.35)
    rect_h = int(h * 0.35)
    x1 = int((w - rect_w) / 2)
    y1 = int((h - rect_h) / 2)
    x2 = x1 + rect_w
    y2 = y1 + rect_h

    # Skin mask
    mask = skin_mask(frame)
    if args.debug_mask:
        cv2.imshow("mask", mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_point = None
    hand_contour = None
    if contours:
        # pick largest contour by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            hand_contour = cnt
            break

    state_text = "NO HAND"
    color = (200,200,200)
    danger_flash = False

    if hand_contour is not None:
        # compute centroid
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        else:
            cx, cy = 0, 0

        # fingertip candidate: point on contour maximizing distance from centroid
        cnt_pts = hand_contour.reshape(-1,2)
        dists = np.linalg.norm(cnt_pts - np.array([cx,cy]), axis=1)
        idx = np.argmax(dists)
        fx, fy = int(cnt_pts[idx][0]), int(cnt_pts[idx][1])
        hand_point = (fx, fy)

        # draw contour and point
        cv2.drawContours(frame, [hand_contour], -1, (0,160,20), 2)
        cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
        cv2.circle(frame, hand_point, 8, (0,0,255), -1)

        # Compute distance to rect boundary
        dist = point_to_rect_distance(fx, fy, x1, y1, x2, y2)

        # thresholds scaled with frame diagonal
        diag = math.hypot(w, h)
        safe_thresh = diag * args.safe_ratio    # far => SAFE if > this
        warn_thresh = diag * args.warning_ratio # WARNING zone if <= safe_thresh and > warn_thresh
        danger_thresh = 0                       # touching/inside => dist == 0 -> DANGER

        if dist == 0:
            state_text = "DANGER"
            color = (0,0,255)
            danger_flash = True
        elif dist <= warn_thresh:
            state_text = "WARNING"
            color = (0,165,255)
        else:
            state_text = "SAFE"
            color = (0,200,0)

        # Draw distance annotation
        cv2.putText(frame, f"dist:{int(dist)} px", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    else:
        # optional fallback: background subtraction or motion detection not implemented here
        state_text = "NO HAND"
        color = (200,200,200)

    # Draw virtual rectangle
    cv2.rectangle(frame, (x1,y1), (x2,y2), (180,180,180), 2)
    # Optionally fill inner region lightly (transparent effect)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (50,50,50), -1)
    alpha = 0.07
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    # Draw state
    cv2.putText(frame, f"STATE: {state_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Flashing danger text
    if state_text == "DANGER":
        # Flashing effect
        t = time.time()
        if int(t*2) % 2 == 0:  # toggle at 2 Hz
            cv2.putText(frame, "DANGER DANGER", (int(w*0.12), int(h*0.90)), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,0,255), 4)
    # FPS calculation
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if cur_time != prev_time else 0.0
    prev_time = cur_time
    fps_smooth = 0.8*fps_smooth + 0.2*fps
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Hand-Boundary POC", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
