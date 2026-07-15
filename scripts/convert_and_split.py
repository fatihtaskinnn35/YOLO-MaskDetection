"""Pascal VOC XML -> YOLO txt donusumu + train/val split.

Girdi:  dataset/raw/extracted/images/*.png, dataset/raw/extracted/annotations/*.xml
Cikti:  dataset/images/{train,val}/*.png
        dataset/labels/{train,val}/*.txt
"""
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "dataset" / "raw" / "extracted" / "images"
RAW_ANNOTS = ROOT / "dataset" / "raw" / "extracted" / "annotations"

OUT_IMAGES = ROOT / "dataset" / "images"
OUT_LABELS = ROOT / "dataset" / "labels"

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

VAL_RATIO = 0.2
SEED = 42


def convert_one(xml_path: Path) -> tuple[str, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_id = CLASS_TO_ID[name]

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return filename, lines


def main():
    xml_files = sorted(RAW_ANNOTS.glob("*.xml"))
    print(f"Toplam {len(xml_files)} XML dosyasi bulundu.")

    random.seed(SEED)
    shuffled = xml_files[:]
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * VAL_RATIO)
    val_set = set(shuffled[:val_count])

    for split in ("train", "val"):
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_LABELS / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    skipped = 0

    for xml_path in xml_files:
        filename, lines = convert_one(xml_path)
        img_path = RAW_IMAGES / filename

        if not img_path.exists():
            skipped += 1
            continue

        split = "val" if xml_path in val_set else "train"
        counts[split] += 1

        shutil.copy2(img_path, OUT_IMAGES / split / filename)

        label_name = Path(filename).stem + ".txt"
        (OUT_LABELS / split / label_name).write_text("\n".join(lines), encoding="utf-8")

    print(f"train: {counts['train']} gorsel, val: {counts['val']} gorsel")
    if skipped:
        print(f"UYARI: {skipped} XML icin karsilik gelen gorsel bulunamadi, atlandi.")
    print("Siniflar:", CLASS_TO_ID)


if __name__ == "__main__":
    main()
