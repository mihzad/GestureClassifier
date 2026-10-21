import cv2
import os
import shutil

# ======== SETTINGS ========
max_w, max_h = 1200, 700  # for display scaling
camera_folder = "../Video"
output_folder = "../Video/cropped_videos"
processed_folder = "../Video/processed_shit"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)


# ======== FUNCTIONS ========

def choose_crop_area(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read first frame.")
        return None

    window_name = "Select crop area (drag a square, press any key to confirm)"
    drawing = False
    ix, iy = -1, -1
    crop_rect = []

    h, w = frame.shape[:2]
    scale = 1
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        frame_display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_display = frame.copy()

    def draw_square(event, x, y, flags, param):
        nonlocal ix, iy, drawing, crop_rect
        img_copy = frame_display.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            dx = x - ix
            dy = y - iy
            side = min(abs(dx), abs(dy))
            x2 = ix + side if dx >= 0 else ix - side
            y2 = iy + side if dy >= 0 else iy - side
            cv2.rectangle(img_copy, (ix, iy), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            dx = x - ix
            dy = y - iy
            side = min(abs(dx), abs(dy))
            x2 = ix + side if dx >= 0 else ix - side
            y2 = iy + side if dy >= 0 else iy - side
            crop_rect[:] = [min(ix, x2), min(iy, y2), side, side]
            cv2.rectangle(img_copy, (ix, iy), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_square)
    cv2.imshow(window_name, frame_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not crop_rect:
        print("No crop selected.")
        return None

    # Scale back to original resolution
    x, y, s, _ = [int(v / scale) for v in crop_rect]
    return (x, y, s, s)


def crop_video(input_path, output_path, crop_rect):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    x, y, w, h = crop_rect
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y:y + h, x:x + w]
        out.write(cropped)

    cap.release()
    out.release()
    print(f"Saved cropped video → {output_path}")


def tag_as_processed(input_vid_path):
    global processed_folder
    os.makedirs(processed_folder, exist_ok=True)

    if not hasattr(tag_as_processed, "scene_dir"):
        scenes = [scene_dir for scene_dir in os.listdir(processed_folder)
                  if os.path.isdir(os.path.join(processed_folder, scene_dir))]
        spc = len(scenes)  # scenes_processed_count
        tag_as_processed.scene_dir = os.path.join(processed_folder, f"scene{spc + 1}")

    os.makedirs(tag_as_processed.scene_dir, exist_ok=True)
    shutil.move(src=input_vid_path, dst=os.path.join(tag_as_processed.scene_dir, os.path.basename(input_vid_path)))

def process_camera_videos():
    video_files = [f for f in os.listdir(camera_folder)
                   if f.lower().endswith((".mp4", ".avi", ".mov"))]

    if not video_files:
        print("No videos found in camera folder.")
        return

    for video_name in video_files:
        input_path = os.path.join(camera_folder, video_name)
        print(f"\nProcessing: {video_name}")

        crop_rect = choose_crop_area(input_path)
        if not crop_rect:
            print(f"No rectangle selected for {video_name}. Skipping it.")
            continue

        if not hasattr(process_camera_videos, "scene_dir"):
            scenes = [scene_dir for scene_dir in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, scene_dir))]
            spc = len(scenes)  # scenes_processed_count
            process_camera_videos.scene_dir = os.path.join(output_folder, f"scene{spc + 1}")

        os.makedirs(process_camera_videos.scene_dir, exist_ok=True)
        output_path = os.path.join(process_camera_videos.scene_dir, f"{video_name}")
        crop_video(input_path, output_path, crop_rect)
        tag_as_processed(input_path)
        print("Done.\n")


if __name__ == "__main__":
    process_camera_videos()
