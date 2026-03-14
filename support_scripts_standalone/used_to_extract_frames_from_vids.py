import cv2
import os
from PIL import Image
import shutil

#========== GLOBAL SETTINGS ==========
frame_size_needed = (224,224)
max_w, max_h = 1200, 700 #for display of cv2, needed for correct cropping

# folder with videos to process, presumably just camera folder
camera_folder = "../Video"

# folder to put frames into after extraction
dataset_folder = "../data/gg"

# where to put vids into after the job`s done, useful for preventing confusions
processed_folder = "../Video/processed_shit"


def extract_frames(video_path, step=-1, warmup=0, fr_retrieve = 16):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if step == -1:
        step = frame_count // fr_retrieve
    frames_to_return = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i >= warmup and i % step == 0 and len(frames_to_return) < fr_retrieve:
            frames_to_return.append(frame)
        i += 1

    return frames_to_return

def enable_crop(input_frames):
    drawing = False
    ix, iy = -1, -1
    crop_rect = []

    #========== SELECT CROP ==========
    def draw_square(event, x, y, flags, param):
        nonlocal ix, iy, drawing, crop_rect

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = img_resized.copy()

            dx = x - ix
            dy = y - iy

            # Determine side length for square
            side = min(abs(dx), abs(dy))

            # Preserve drag direction (corner)
            if dx >= 0 and dy >= 0:
                x2, y2 = ix + side, iy + side
            elif dx < 0 and dy >= 0:
                x2, y2 = ix - side, iy + side
            elif dx >= 0 and dy < 0:
                x2, y2 = ix + side, iy - side
            else:  # dx < 0 and dy < 0
                x2, y2 = ix - side, iy - side

            cv2.rectangle(img_copy, (ix, iy), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

            dx = x - ix
            dy = y - iy
            side = min(abs(dx), abs(dy))

            if dx >= 0 and dy >= 0:
                x2, y2 = ix + side, iy + side
            elif dx < 0 and dy >= 0:
                x2, y2 = ix - side, iy + side
            elif dx >= 0 and dy < 0:
                x2, y2 = ix + side, iy - side
            else:
                x2, y2 = ix - side, iy - side

            crop_rect[:] = [min(ix, x2), min(iy, y2), side, side]


    if len(input_frames) == 0:
        print("Could not read first frame - there is none.")
        return input_frames

    sample_img = input_frames[0]
    window_name = f"Select the rectangle you want your frames to be cropped to:"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_square)

    h, w = sample_img.shape[:2]
    scale = 1 # will be changed if img needs resizing

    if w > max_w or h > max_h:
        img_cpy = sample_img.copy()
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img_cpy, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = sample_img

    cv2.imshow(window_name, img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not crop_rect or crop_rect[-1] == 0 or crop_rect[-2] == 0:
        print("No crop selected.")
        return input_frames

    # ========== ENABLE CROP PER-FRAME ==========
    x, y, w, h = [round(v / scale) for v in crop_rect]


    cropped = [frame[y:y + h, x:x + w] for frame in input_frames]
    return cropped

def tag_as_processed(input_vid_path):
    global processed_folder
    os.makedirs(processed_folder, exist_ok=True)

    if not hasattr(tag_as_processed, "scene_dir"):
        scenes = [scene_dir for scene_dir in os.listdir(processed_folder)
                  if os.path.isdir(os.path.join(processed_folder, scene_dir))]
        spc = len(scenes) #scenes_processed_count
        tag_as_processed.scene_dir = os.path.join(processed_folder, f"scene{spc + 1}")

    os.makedirs(tag_as_processed.scene_dir, exist_ok=True)
    shutil.move(src=input_vid_path, dst=os.path.join(tag_as_processed.scene_dir, os.path.basename(input_vid_path)))

def read_camera_to_dataset(output_data_root_dir):
    global camera_folder

    vid_names = sorted([v_name for v_name in os.listdir(camera_folder) if
                   not os.path.isdir(os.path.join(camera_folder, v_name))
                   and ".ini" not in v_name]) #remove desktop.ini sneaking in


    #highly configurable: any letter sequence can be entered
    # just be sure your videos in camera_folder match EXACTLY (order matters).
    class_names = [
        'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є',
        'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л',
        'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
        'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я'
    ]

    if len(vid_names) != len(class_names):
        print(f"gotta need {len(class_names)} gesture vids. got: {len(vid_names)}")
        return

    for class_idx in range(len(class_names)):
        class_path = os.path.join(output_data_root_dir, class_names[class_idx])
        os.makedirs(class_path, exist_ok=True)

        vid_dirs_existed = sorted([d for d in os.listdir(class_path)
                                   if os.path.isdir(os.path.join(class_path, d))])

        #creating new folder - a bit complicated to be sure it wont overwrite smth.
        vids_count = len(vid_dirs_existed)
        while True:
            output_vid_path = os.path.join(class_path, f"vid{vids_count + 1}_test")
            if not os.path.exists(output_vid_path):
                break
            vids_count += 1
        os.makedirs(output_vid_path, exist_ok=False)


        input_vid_path = os.path.join(camera_folder, vid_names[class_idx])

        print(f"video '{class_names[class_idx]}': extracting frames...")
        frames = extract_frames(video_path=input_vid_path) #gotta get 16 frames
        print(f"video '{class_names[class_idx]}': enabling crop...")
        frames = enable_crop(input_frames=frames)
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        print(f"video '{class_names[class_idx]}': saving...")
        for i, f in enumerate(frames_rgb):
            img = Image.fromarray(f)
            img = img.resize(size=frame_size_needed)
            img_path = os.path.join(output_vid_path, f"frame_{(i+1):05d}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path, format="JPEG")
            else:
                print(f"video '{class_names[class_idx]}': folder_already_exists.\n")
                break

        tag_as_processed(input_vid_path=input_vid_path)
        print(f"video '{class_names[class_idx]}': done.\n")


if __name__ == "__main__":
    read_camera_to_dataset(dataset_folder)