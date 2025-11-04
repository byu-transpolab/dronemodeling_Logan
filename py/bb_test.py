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
    skipped_file_path = out_folder / "files_skipped.txt"
    with open(skipped_file_path, "w") as f:
        for name in skipped_files:
            f.write(f"{name}\n")

    print(f"Done! CSV saved at: {output_csv}")
    print(f"Skipped files saved at: {skipped_file_path}")


normalized_bb_dir = "/Users/willicon/Desktop/seed80_2025_11_04_133501/normalized_bb"
yolo_dir = "/Users/willicon/Desktop/dronemodeling_Logan/runs/detect/predict/labels"


match_normalized_to_yolo(normalized_bb_dir, yolo_dir)


