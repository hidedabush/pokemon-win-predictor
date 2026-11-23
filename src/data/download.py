"""Download Gen 9 OU replays from Hugging Face.        Owner: Alexander

Output: data/raw/gen9ou_raw.parquet  (columns as in the HF dataset)

TODO
- [ ] Look at the dataset's file list on Hugging Face and download only the parquet shards needed
      (the full dataset is ~34M rows across all formats; we only want formatid == "gen9ou").
- [ ] Filter to cfg["data"]["formatid"] as early as possible (polars/pyarrow filter on read).
- [ ] Keep a bit more than sample_size (e.g. 3x) because parsing/filtering will drop rows.
- [ ] Save to data/raw/. Print how many rows you kept.
"""

from src.utils import ROOT, load_config


def download(cfg: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    cfg = load_config()
    (ROOT / cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    download(cfg)
