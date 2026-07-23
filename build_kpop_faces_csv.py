"""
Builds train/test CSVs from data/Kpop_Faces, which is organized as
<group>/<person>/<image>. Labels are the person's name; group folders are
only used to disambiguate people who share a stage name (e.g. two different
idols both going by "Yuna").

Outputs data/kpop_faces_train.csv and data/kpop_faces_test.csv with columns
file_name,name, where file_name is relative to data/Kpop_Faces (e.g.
"blackpink/jisoo/198470_njcI.png").
"""
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DATA_DIR = Path("data/Kpop_Faces")
TRAIN_CSV = Path("data/kpop_faces_train.csv")
TEST_CSV = Path("data/kpop_faces_test.csv")
TEST_FRACTION = 0.1
SEED = 42


def collect_images():
    """Maps person name -> list of (group, image_path)."""
    by_person = defaultdict(list)
    for group_dir in sorted(DATA_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for person_dir in sorted(group_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            images = [
                p for p in sorted(person_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXTS
            ]
            if images:
                by_person[person_dir.name].append((group_dir.name, images))
    return by_person


def resolve_labels(by_person):
    """Returns list of (label, image_path) pairs, disambiguating names that
    belong to more than one group by prefixing them with the group name."""
    rows = []
    for person, group_images in by_person.items():
        ambiguous = len(group_images) > 1
        for group, images in group_images:
            label = f"{group}_{person}" if ambiguous else person
            for img in images:
                rows.append((label, img))
    return rows


def split_train_test(rows):
    by_label = defaultdict(list)
    for label, img in rows:
        by_label[label].append(img)

    rng = random.Random(SEED)
    train_rows, test_rows = [], []
    for label, images in by_label.items():
        images = images[:]
        rng.shuffle(images)
        n_test = max(1, round(len(images) * TEST_FRACTION)) if len(images) > 1 else 0
        n_test = min(n_test, len(images) - 1) if len(images) > 1 else 0
        test_imgs = images[:n_test]
        train_imgs = images[n_test:]
        train_rows += [(img, label) for img in train_imgs]
        test_rows += [(img, label) for img in test_imgs]
    return train_rows, test_rows


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write("file_name,name\n")
        for img, label in rows:
            rel_path = img.relative_to(DATA_DIR)
            f.write(f"{rel_path},{label}\n")


def main():
    by_person = collect_images()
    rows = resolve_labels(by_person)
    train_rows, test_rows = split_train_test(rows)

    write_csv(TRAIN_CSV, train_rows)
    write_csv(TEST_CSV, test_rows)

    n_labels = len({label for label, _ in rows})
    print(f"{n_labels} people, {len(train_rows)} train images, {len(test_rows)} test images")
    print(f"Wrote {TRAIN_CSV} and {TEST_CSV}")


if __name__ == "__main__":
    main()
