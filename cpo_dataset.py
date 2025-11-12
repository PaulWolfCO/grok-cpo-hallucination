# cpo_dataset.py
from pathlib import Path
from datasets import Dataset

def load_triplets(src_path, good_path, bad_path):
    src = Path(src_path).read_text().strip().splitlines()
    good = Path(good_path).read_text().strip().splitlines()
    bad = Path(bad_path).read_text().strip().splitlines()
    assert len(src) == len(good) == len(bad)
    return Dataset.from_dict({"src": src, "good": good, "bad": bad})

if __name__ == "__main__":
    dataset = load_triplets(
        "examples/source.txt",
        "examples/good_fr.txt",
        "examples/bad_fr.txt",
    )
    dataset.save_to_disk("cpo_triplets")
    print(f"Saved {len(dataset)} triplets")