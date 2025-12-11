from pathlib import Path
import csv

def bbox_iou(boxA, boxB):
    """Compute IoU between two boxes (x_min, y_min, x_max, y_max)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def yolo_to_bbox(yolo_line):
    """Convert YOLO line to (class_id, x_min, y_min, x_max, y_max)."""
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO line: {yolo_line}")
    class_id, x_c, y_c, w, h = map(float, parts)
    x_min = x_c - w/2
    y_min = y_c - h/2
    x_max = x_c + w/2
    y_max = y_c + h/2
    return int(class_id), (x_min, y_min, x_max, y_max)

def normalize_latlon_bboxes(image_size_bb_dir, building_bb_dir, output_dir=None):
    """
    Normalize building bounding boxes relative to true bounding boxes.
    Each bbox is in lat/lon format: west,south,east,north.
    Output values are normalized (0-1) relative to the true imaged sized bbox.
    """

    image_size_bb_dir = Path(image_size_bb_dir)
    building_bb_dir = Path(building_bb_dir)
    output_dir = Path(output_dir) if output_dir else building_bb_dir.parent / "normalized_bb"
    output_dir.mkdir(parents=True, exist_ok=True)

    count_written = 0

    for req_file in building_bb_dir.glob("*.txt"):
        name = req_file.name
        image_size_file = image_size_bb_dir / name

        if not image_size_file.exists():
            print(f"Skipping {name} — no matching file in building_bb folder.")
            continue

        try:
            # Read requested (building) bbox
            with open(req_file) as f:
                west_r, south_r, east_r, north_r = map(float, f.read().strip().split(","))

            # Read true bbox
            with open(image_size_file) as f:
                west_t, south_t, east_t, north_t = map(float, f.read().strip().split(","))

            # Avoid division by zero
            if east_t == west_t or north_t == south_t:
                print(f"Invalid imaged sized bbox for {name}, skipping.")
                continue

            # Normalize relative to true bounding box
            norm_w = (west_r - west_t) / (east_t - west_t)
            norm_e = (east_r - west_t) / (east_t - west_t)
            norm_s = (south_r - south_t) / (north_t - south_t)
            norm_n = (north_r - south_t) / (north_t - south_t)

            # Save normalized bbox
            out_file = output_dir / name
            with open(out_file, "w") as f:
                f.write(f"{norm_w:.6f},{norm_s:.6f},{norm_e:.6f},{norm_n:.6f}")

            count_written += 1
            print(f"Saved normalized bbox: {out_file}")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    print(f"\nFinished! {count_written} normalized bbox files written to: {output_dir}")

def match_normalized_to_yolo(normalized_bb_dir, yolo_dir, output_csv=None):
    normalized_bb_dir = Path(normalized_bb_dir)
    yolo_dir = Path(yolo_dir)
    
    # If no output CSV path is given, create a folder one level up
    if output_csv is None:
        parent_dir = normalized_bb_dir.parent
        out_folder = parent_dir / "matched_output"
        out_folder.mkdir(parents=True, exist_ok=True)
        output_csv = out_folder / "matched_classes.csv"
    else:
        output_csv = Path(output_csv)
        out_folder = output_csv.parent
        out_folder.mkdir(parents=True, exist_ok=True)

    results = []
    skipped_files = []

    for norm_file in normalized_bb_dir.glob("*.txt"):
        image_name = norm_file.stem
        yolo_file = yolo_dir / f"{image_name}.txt"

        if not yolo_file.exists():
            skipped_files.append(image_name)
            continue

        # Load normalized bbox
        norm_vals = list(map(float, norm_file.read_text().strip().split(",")))
        norm_bbox = (norm_vals[0], norm_vals[1], norm_vals[2], norm_vals[3])

        best_iou = -1
        best_class = None

        # Read YOLO annotations
        with open(yolo_file) as f:
            lines = f.readlines()
            for line in lines:
                class_id, yolo_bbox = yolo_to_bbox(line)
                iou = bbox_iou(norm_bbox, yolo_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_class = class_id

        if best_class is not None:
            results.append((image_name, best_class))
        else:
            skipped_files.append(image_name)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name","class_id"])
        writer.writerows(results)

    # Write skipped files
    skipped_file_path = out_folder / "images_skipped.txt"
    with open(skipped_file_path, "w") as f:
        for name in skipped_files:
            f.write(f"{name}\n")

    print(f"Done! CSV saved at: {output_csv}")
    print(f"Skipped files saved at: {skipped_file_path}. These images did not have predictions")

def Match_yolo_output_to_footprints(image_folder, yolo_dir, output):
    """
    Takes the predicted images and corrdinates the building type with the bulding footprint ID

    Arg
    Image_folder: path to file with the images that the yolo model ran its prediction on. This file must also contain the bounding boxes for the building and the image in lat-long
    yolo_dir: Path to yolo model prediction labels
    output: Type of output desired. (EXPLAIN OUTPUT OPTIONS)
    
    """
    image_size_bb_folder = f"{image_folder}/image_size_bb"
    building_bb_folder = f"{image_folder}/building_bb"
    normalized_bb_dir = f"{image_folder}/normalized_bb"


    normalize_latlon_bboxes(image_size_bb_folder, building_bb_folder)
    match_normalized_to_yolo(normalized_bb_dir, yolo_dir, output)



image_folder="/Users/willicon/Desktop/Buildings_seed25_2025_12_01_150531_stgeorge_kmeans_cluster.geojson"
yolo_dir = "/Users/willicon/Desktop/dronemodeling_Logan/runs/detect/stgeorge_run9/labels"

Match_yolo_output_to_footprints(image_folder, yolo_dir, output=None)