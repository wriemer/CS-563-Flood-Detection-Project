import os
import shutil
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def combine_datasets(kaggle_dir, floodnet_dir, output_dir, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    classes = ['Flooded', 'Non-Flooded']

    # Create output directories:
    #     train
    #       Flooded
    #       Non-Flooded
    #     test
    #       Flooded
    #       Non-Flooded
    for split in ['train', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Kaggle already split into train/test -> use existing ratio to split FloodNet into train/test
    kaggle_train_csv = os.path.join(kaggle_dir, 'train.csv')
    kaggle_test_csv = os.path.join(kaggle_dir, 'test.csv')

    kaggle_train_df = pd.read_csv(kaggle_train_csv)
    kaggle_test_df = pd.read_csv(kaggle_test_csv)

    kaggle_train_ratio = len(kaggle_train_df) / (len(kaggle_train_df) + len(kaggle_test_df))
    print(f"Kaggle train/test ratio: {kaggle_train_ratio:.2f} / {1-kaggle_train_ratio:.2f}")

    # Read in + split FloodNet dataset
    print("Processing FloodNet dataset...")
    floodnet_base = os.path.join(floodnet_dir, 'Train', 'Labeled')
    floodnet_images = []

    for folder, label_name in [('Flooded', 'Flooded'), ('Non-Flooded', 'Non-Flooded')]:
        img_dir = os.path.join(floodnet_base, folder, 'image')
        if not os.path.exists(img_dir):
            print(f"Missing FloodNet folder: {img_dir}")
            continue
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                floodnet_images.append((os.path.join(img_dir, img_file), label_name))

    # Split FloodNet using Kaggle's ratio
    train_imgs, test_imgs = train_test_split(
        floodnet_images,
        train_size=kaggle_train_ratio,
        stratify=[l for _, l in floodnet_images],
        random_state=seed
    )

    for img_list, split_name in [(train_imgs, 'train'), (test_imgs, 'test')]:
        for src_path, label in tqdm(img_list, desc=f"Copying FloodNet {split_name}"):
            dest_path = os.path.join(output_dir, split_name, label, f"floodnet_{os.path.basename(src_path)}")
            shutil.copy(src_path, dest_path)

    # Copy over images from Kaggle dataset
    print("\nProcessing Kaggle dataset...")

    def copy_from_csv(df, split_name):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Kaggle {split_name}"):
            img_file = row['Image ID']
            # Determine label based on columns: Flooded or Normal
            if row['Flooded'] == 1:
                label_dir = 'Flooded'
            else:
                label_dir = 'Non-Flooded'
            src_path = os.path.join(kaggle_dir, split_name, img_file)
            if not os.path.exists(src_path):
                continue
            dest_path = os.path.join(output_dir, split_name, label_dir, f"kaggle_{img_file}")
            shutil.copy(src_path, dest_path)

    copy_from_csv(kaggle_train_df, 'train')
    copy_from_csv(kaggle_test_df, 'test')

    print("\nCombined dataset created at:", output_dir)
    print(f"Train images: {len(os.listdir(os.path.join(output_dir, 'train', 'Flooded')))+len(os.listdir(os.path.join(output_dir, 'train', 'Non-Flooded')))}")
    print(f"Test images: {len(os.listdir(os.path.join(output_dir, 'test', 'Flooded')))+len(os.listdir(os.path.join(output_dir, 'test', 'Non-Flooded')))}")


if __name__ == "__main__":
    kaggle_source = "Kaggle-Dataset"
    floodnet_source = "Floodnet-Dataset"
    output_path = "Comb-Dataset"

    combine_datasets(kaggle_source, floodnet_source, output_path)
