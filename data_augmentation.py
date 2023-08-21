import os
from PIL import Image

def rotate_and_save(img_rgb, angle, filename, folder):
    base_name, ext = os.path.splitext(filename)
    img_rotated = img_rgb.rotate(angle)
    img_rotated_path = os.path.join(folder, f"{base_name}_rotated_{angle}{ext}")
    img_rotated.save(img_rotated_path)
    print(f"Saved {img_rotated_path}")

def augment_images_in_folder(folder):
    # Allowed extensions tuple
    allowed_extensions = ('.webp', '.jpg', '.png')
    
    for filename in os.listdir(folder):
        if filename.endswith(allowed_extensions):  
            image_path = os.path.join(folder, filename)
            
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                base_name, ext = os.path.splitext(filename)

                # Mirror the image (horizontal flip) and save it
                img_mirrored = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                img_mirrored_path = os.path.join(folder, f"{base_name}_mirrored{ext}")
                img_mirrored.save(img_mirrored_path)
                print(f"Saved {img_mirrored_path}")

                # List of angles for rotation
                angles = [-15, 15, 90, 180, 270]

                # Perform rotations only for the original image
                for angle in angles:
                    rotate_and_save(img_rgb, angle, filename, folder)

if __name__ == "__main__":
    denominations = [
        "100", "1000", "10000", "200", "50", "500", "5000"
    ]

    # Retrieve the directory of this script file
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the Images directory relative to the script
    base_folder = os.path.join(script_directory, "Images")

    for denomination in denominations:
        denomination_folder = os.path.join(base_folder, denomination)
        augment_images_in_folder(denomination_folder)
