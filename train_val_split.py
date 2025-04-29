# threaded_split.py
import os, shutil, random, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def split_subject(subject_dir: str, train_root: str, val_root: str,
                  train_ratio: float):
    """
    Copy images from one subject directory to train/val sub-folders.
    """
    subject = os.path.basename(subject_dir)

    images = os.listdir(subject_dir)
    if not images:                     # skip empty folders
        return 0, 0

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]

    # prepare destination dirs once
    train_dst = os.path.join(train_root, subject)
    val_dst   = os.path.join(val_root,   subject)
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst,   exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(subject_dir, img),
                     os.path.join(train_dst,  img))

    for img in val_imgs:
        shutil.copy2(os.path.join(subject_dir, img),
                     os.path.join(val_dst,    img))

    return len(train_imgs), len(val_imgs)


def threaded_split(dataset_path: str, train_path: str, val_path: str,
                   train_ratio: float = 0.9, num_workers: int | None = None):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path,   exist_ok=True)

    subjects = [os.path.join(dataset_path, d)
                for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))]

    total_train = total_val = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(split_subject, s, train_path, val_path,
                               train_ratio): s for s in subjects}

        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="Splitting", unit="subject"):
            tr, vl = f.result()
            total_train += tr
            total_val   += vl

    print(f"Dataset split complete!  "
          f"{total_train} images in train, {total_val} images in val.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Root folder containing per-identity sub-folders")
    ap.add_argument("--train", type=str, help="Train folder")
    ap.add_argument("--val", type=str, help="Val folder")
    ap.add_argument("--out", default="ms1m-retinaface", help="Output root")
    ap.add_argument("--train_ratio", type=float, default=0.9,
                    help="Fraction of images that go to train set")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Thread count (default: CPU cores)")
    args = ap.parse_args()

    threaded_split(
        dataset_path=args.dataset,
        train_path = args.train,
        val_path   = args.val,
        train_ratio=args.train_ratio,
        num_workers=args.workers
    )