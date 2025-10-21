import os
from pathlib import Path
from PIL import Image

def mirror_dataset(root_dir):
    root = Path(root_dir)

    # Traverse all subsubfolders
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            for subsubfolder in subfolder.iterdir():
                if subsubfolder.is_dir():
                    # Define output folder
                    output_folder = subfolder / f"{subsubfolder.name}_mirror"
                    output_folder.mkdir(parents=True, exist_ok=True)

                    # Process images in subsubfolder
                    for img_path in subsubfolder.glob("*.jpg"):
                        try:
                            img = Image.open(img_path)
                            mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)

                            # Save with the same name in output folder
                            mirrored.save(output_folder / img_path.name)
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    dataset_dir = "../data/test"
    mirror_dataset(dataset_dir)
    print("Mirrored dataset created.")