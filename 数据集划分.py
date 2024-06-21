import os
import random
from shutil import copyfile

def split_dataset(image_folder, txt_folder, output_folder, split_ratio=(0.8, 0.1, 0.1)):
    # Ensure output folders exist
    for dataset in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(output_folder, dataset, 'images')):
            os.makedirs(os.path.join(output_folder, dataset, 'images'))
        if not os.path.exists(os.path.join(output_folder, dataset, 'txt')):
            os.makedirs(os.path.join(output_folder, dataset, 'txt'))

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    num_images = len(image_files)
    num_train = int(split_ratio[0] * num_images)
    num_val = int(split_ratio[1] * num_images)

    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    # Copy images to respective folders
    for dataset, images_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        for image_file in images_list:
            image_path = os.path.join(image_folder, image_file)
            copyfile(image_path, os.path.join(output_folder, dataset, 'images', image_file))
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            txt_path = os.path.join(txt_folder, txt_file)

            # Copy corresponding txt file if exists
            if os.path.exists(txt_path):
                copyfile(txt_path, os.path.join(output_folder, dataset, 'txt', txt_file))

if __name__ == "__main__":
    image_folder_path = "D:\Code\Python\well\images"
    txt_folder_path = "D:\Code\Python\well\labels"
    output_dataset_path = "D:\Code\Python\well\data"

    split_dataset(image_folder_path, txt_folder_path, output_dataset_path)
