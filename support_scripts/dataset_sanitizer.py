import os
import shutil
import re
from rich import print as rich_print
def sanitize(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for cls_name in classes:
        current_video_name = None
        current_video_frames = []
        video_counter = 1
        class_dir = os.path.join(root_dir, cls_name)
        files_available = [f for f in sorted(os.listdir(class_dir))
                           if not os.path.isdir(  os.path.join(class_dir, f)  )]
        for filename in files_available:

            vid_name = filename.split(sep='.')[0]

            if current_video_name is None:
                current_video_name = vid_name
                current_video_frames.append(filename)

            elif current_video_name == vid_name:
                current_video_frames.append(filename)
            else:
                move_to_new_videodir(root_dir=class_dir, files_to_move=current_video_frames, video_counter=video_counter)
                current_video_name = vid_name
                current_video_frames = [filename]
                video_counter += 1

        #move the last ones
        move_to_new_videodir(root_dir=class_dir, files_to_move=current_video_frames, video_counter=video_counter)


def move_to_new_videodir(root_dir, files_to_move, video_counter):

    new_video_dir = os.path.join(root_dir, f"vid{video_counter}")

    os.makedirs(new_video_dir, exist_ok=True)

    for f in files_to_move:
        shutil.move(src=os.path.join(root_dir, f), dst=os.path.join(new_video_dir, f) )






def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
def red(text):
    return f"\033[91m{text}\033[0m"  # ANSI code for red
def analyze(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for cls_name in classes:
        class_dir = os.path.join(root_dir, cls_name)
        videodirs = sorted([v for v in (os.listdir(class_dir))
                           if os.path.isdir(os.path.join(class_dir, v))], key=natural_key)
        class_videoinfo = []
        for v in videodirs:
            vid_dir = os.path.join(class_dir, v)
            files_available = [f for f in os.listdir(vid_dir)
                               if not os.path.isdir(os.path.join(vid_dir, f))]
            frame_count = len(files_available)
            count_str = f"[red]{frame_count}[/red]" if frame_count < 16 else str(frame_count)
            class_videoinfo.append((v, count_str))

        rich_print(f"class '{cls_name}': \n{class_videoinfo}\n\n")



analyze(root_dir="D:/PyCharm_Community_Edition_2023.2.2/_Solutions/GestureClassifier/testing/train")