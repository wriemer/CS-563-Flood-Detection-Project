import os
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
import rasterio

def save_sen12_ndwi_pngs(sen12_root, output_dir, max_items=None, debug=False):
    sensor = "s2"
    label_root = os.path.join(
        sen12_root, f'sen12floods_{sensor}_labels', f'sen12floods_{sensor}_labels'
    )
    source_root = os.path.join(
        sen12_root, f'sen12floods_{sensor}_source', f'sen12floods_{sensor}_source'
    )

    os.makedirs(output_dir, exist_ok=True)

    items = sorted(glob(os.path.join(label_root, f"sen12floods_{sensor}_labels_*")))
    if max_items is not None and max_items < len(items):
        items = items[:int(max_items / 2)] + items[-int(max_items / 2):]

    print(len(items), "items found.")
    train_records, test_records = [], []

    for label_dir in tqdm(items, desc=f"Processing SEN12 {sensor.upper()}"):
        stac_path = os.path.join(label_dir, "stac.json")
        geojson_path = os.path.join(label_dir, "labels.geojson")
        if not (os.path.exists(stac_path) and os.path.exists(geojson_path)):
            continue

        # Extract id num from folder structure (sen12floods_s2_labels_0011_YYYY_MM_DD) => 0032 for example is train, 32 would be test
        basename = os.path.basename(label_dir)
        parts = basename.split('_')
        if len(parts) < 5:
            continue
        id_part = parts[3]
        if not id_part.isdigit():
            continue

        subset = 'train' if len(id_part) == 4 else 'test'

        # Load metadata (label, etc)
        with open(stac_path, "r") as f:
            stac_data = json.load(f)
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)

        # Only use full coverage images
        if not geojson_data["properties"].get("FULL-DATA-COVERAGE", False):
            if debug:
                print(f"Skipping {label_dir}: FULL-DATA-COVERAGE=False")
            continue

        label = "Flooded" if geojson_data["properties"]["FLOODING"] else "Non-Flooded"

        # Find corresponding source folder
        source_folder = None
        for link in stac_data.get("links", []):
            if link["rel"] == "source":
                source_folder = os.path.normpath(
                    os.path.join(source_root, os.path.basename(os.path.dirname(link["href"])))
                )
                break

        if not (source_folder and os.path.exists(source_folder)):
            if debug:
                print(f"Skipping {label_dir}: missing source folder")
            continue

        # Locate green + nir bands
        files = os.listdir(source_folder)
        band_names = {'b03': None, 'b08': None}
        for f in files:
            low = f.lower()
            if 'b03' in low and low.endswith('.tif'):
                band_names['b03'] = os.path.join(source_folder, f)
            elif 'b08' in low and low.endswith('.tif'):
                band_names['b08'] = os.path.join(source_folder, f)

        if not all(band_names.values()):
            if debug:
                print(f"Skipping {source_folder}: missing B03 or B08")
            continue

        # Read bands
        with rasterio.open(band_names['b03']) as src:
            green = src.read(1).astype(np.float32)
        with rasterio.open(band_names['b08']) as src:
            nir = src.read(1).astype(np.float32)

        # Compute NDWI
        ndwi = (green - nir) / (green + nir + 1e-6)
        ndwi = np.clip(ndwi, -1, 1)

        # Scale bands to 0–255
        def scale_band(band):
            band = np.nan_to_num(band)
            mask = band != 0
            if mask.sum() == 0:
                return np.zeros_like(band, dtype=np.uint8)
            minv, maxv = band[mask].min(), band[mask].max()
            scaled = np.zeros_like(band, dtype=np.uint8)
            scaled[mask] = ((band[mask] - minv) / (maxv - minv + 1e-6) * 255).astype(np.uint8)
            return scaled

        green_scaled = scale_band(green)
        nir_scaled = scale_band(nir)
        ndwi_scaled = scale_band(ndwi)

        # Stack as 3 channel image
        arr = np.stack([green_scaled, nir_scaled, ndwi_scaled], axis=-1)
        img = Image.fromarray(arr)

        # Save image
        save_dir = os.path.join(output_dir, subset, label)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{os.path.basename(source_folder)}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)

        record = [save_path, label]
        if subset == 'train':
            train_records.append(record)
        else:
            test_records.append(record)

    train_df = pd.DataFrame(train_records, columns=["Image Path", "Label"])
    test_df = pd.DataFrame(test_records, columns=["Image Path", "Label"])

    return train_df, test_df


if __name__ == "__main__":
    sen12_root = "C:/Users/willr/Classes/Fall 2025/CS-563/Project/Data/Sen12-Dataset/sen12flood"
    output_dir = "C:/Users/willr/Classes/Fall 2025/CS-563/Project/Data/Sen12-Updated"
    train_df, test_df = save_sen12_ndwi_pngs(sen12_root, output_dir, max_items=None, debug=True)
    print("Train DataFrame:")
    print(train_df.head())
    print("Test DataFrame:")
    print(test_df.head())