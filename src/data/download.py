"""Download Gen 9 OU replays from Hugging Face.        Owner: Alexander

Output: data/raw/gen9ou_slim.parquet, one row per battle, with columns
    battle_id, upload_time, players, rating, log_slim

The full dataset is ~73 GB (34M replays, every format), but it is split into one or more
parquet files per format, e.g. "[Gen 9] OU_part1.parquet" ... "[Gen 9] OU_part13.parquet"
(~6 GB in total). We only read those files, and DuckDB streams them straight from Hugging
Face (hf://...) without saving them to disk. Each log is cut down to the only lines we
need, so the output is small:
    |player|p1|<name>|<avatar>|<rating>
    |poke|p1|<Species>, L<level>, <gender>|
    |win|<name>

Usage (all settings are in the `data:` section of config.yaml):
    python -m src.data.download --check      # cheap: list files, count rows per formatid
    python -m src.data.download              # filter + slim -> data/raw/gen9ou_slim.parquet
    python -m src.data.download --inspect    # examples + the checks from data/README.md
    python -m src.data.download --per-file   # fallback: download one file at a time

Hugging Face token: set HF_TOKEN or run `hf auth login` (in Colab, an HF_TOKEN
secret also works). Without a token it still works but is rate-limited.
"""

import argparse
import os
import re
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from tqdm import tqdm

from src.utils import ROOT, load_config, to_id

# A whole line that starts with |player|, |poke| or |win|. (?m) makes ^ match at every
# line start, so a chat message containing "|win|" can't fake a winner.
SLIM_REGEX = r"(?m)^\|(?:player|poke|win)\|[^\r\n]*"


def slim_sql(column: str) -> str:
    """SQL expression that keeps only the |player|, |poke| and |win| lines of a log."""
    return f"array_to_string(regexp_extract_all({column}, '{SLIM_REGEX}'), chr(10))"


def glob_escape(path: str) -> str:
    """DuckDB treats [ ] * ? in file paths as glob patterns. Make them literal.

    "[Gen 9] OU_part1.parquet" would otherwise mean "one of G,e,n,space,9 then ' OU...'".
    """
    return re.sub(r"([\[*?])", r"[\1]", path)


def min_upload_time(cfg: dict) -> int:
    """cfg min_upload_date ("YYYY-MM-DD", UTC) as unix seconds; 0 if not set."""
    d = cfg["data"].get("min_upload_date")
    if not d:
        return 0
    day = date.fromisoformat(str(d))
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def slim_file(cfg: dict) -> Path:
    return ROOT / cfg["data"]["raw_dir"] / cfg["data"]["slim_file"]


# ---------------------------------------------------------------- Hugging Face access


def list_files(cfg: dict, verbose: bool = False) -> list[str]:
    """Return the parquet files to read (repo paths).

    Uses only the files whose name is the format, e.g. "[Gen 9] OU_part3.parquet" for
    formatid "gen9ou". If none match, returns [] and the caller falls back to hf_path_glob.
    """
    files = [
        f
        for f in HfApi().list_repo_tree(
            cfg["data"]["hf_dataset"], repo_type="dataset", recursive=True, token=get_token()
        )
        if isinstance(f, RepoFile) and f.path.endswith(".parquet")
    ]
    print(f"{len(files)} parquet files in the dataset, {sum(f.size for f in files) / 1e9:.1f} GB")

    # "[Gen 9] OU_part3.parquet" -> "[Gen 9] OU" -> "gen9ou"
    fmt = cfg["data"]["formatid"]
    ours = [f for f in files if to_id(re.sub(r"_part\d+$", "", Path(f.path).stem)) == fmt]

    for f in files if verbose else ours:
        mark = "*" if f in ours else " "
        print(f"  {mark} {f.size / 1e6:8.1f} MB  {f.path}")
    if ours:
        print(f"Using {len(ours)} files for {fmt!r}, {sum(f.size for f in ours) / 1e9:.2f} GB")
    else:
        print(f"WARNING: no file names match {fmt!r}; reading hf_path_glob from config.yaml")
    return sorted(f.path for f in ours)


def hf_sources(cfg: dict, paths: list[str]) -> list[str]:
    """hf:// URLs DuckDB should read: the format's own files, else the config glob."""
    base = f"hf://datasets/{cfg['data']['hf_dataset']}/"
    if paths:
        return [base + glob_escape(p) for p in paths]
    return [base + cfg["data"]["hf_path_glob"]]


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection that can read hf:// paths, logged in if a token is available."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET preserve_insertion_order = false")  # lets COPY stream with less RAM
    token = get_token()  # HF_TOKEN env var, `hf auth login` cache, or Colab secret
    if token:
        con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN $token)", {"token": token})
        print("Using your Hugging Face token.")
    else:
        print(
            "WARNING: no Hugging Face token found. Anonymous access is rate-limited per IP "
            "and can fail partway. Fix: `hf auth login` (or set HF_TOKEN)."
        )
    return con


# ---------------------------------------------------------------- the three steps


def check_formats(con: duckdb.DuckDBPyConnection, src: list[str], min_time: int = 0) -> None:
    """Count rows per formatid. Reads only the formatid and uploadtime columns (cheap)."""
    rows = con.execute(
        """
        SELECT formatid, count(*) AS n,
               to_timestamp(min(uploadtime))::DATE AS first,
               to_timestamp(max(uploadtime))::DATE AS last,
               count(*) FILTER (WHERE uploadtime >= $min_time) AS n_after_min_date
        FROM read_parquet($src)
        GROUP BY formatid
        ORDER BY n DESC
        """,
        {"src": src, "min_time": min_time},
    ).fetchall()
    print(f"\n{'formatid':<30}{'rows':>12}  {'first':<12}{'last':<12}{'>= min date':>12}")
    for fmt, n, first, last, n_new in rows:
        print(f"{fmt:<30}{n:>12,}  {first!s:<12}{last!s:<12}{n_new:>12,}")


def slim_query(limit: int | None) -> str:
    """SELECT that filters to one format and date range and slims each log.

    Parameters: $src (list of files), $formatid, $min_time.
    """
    sql = f"""
        SELECT id AS battle_id, uploadtime AS upload_time, players, rating,
               {slim_sql("log")} AS log_slim
        FROM read_parquet($src)
        WHERE formatid = $formatid AND uploadtime >= $min_time
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return sql


def copy_to_parquet(con, query: str, params: dict, out: Path) -> int:
    """Run `query` and write the result straight to `out` (never loaded into Python).

    Writes to a .tmp file first so a crash never leaves a half-written output.
    Returns the number of rows written.
    """
    tmp = out.with_name(out.name + ".tmp").as_posix().replace("'", "''")
    n = con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT parquet)", params).fetchone()[0]
    os.replace(out.with_name(out.name + ".tmp"), out)
    return n


def download(cfg: dict) -> None:
    """Stream the format's replays from Hugging Face, slim the logs, write one parquet."""
    out = slim_file(cfg)
    con = connect()
    params = {
        "src": hf_sources(cfg, list_files(cfg)),
        "formatid": cfg["data"]["formatid"],
        "min_time": min_upload_time(cfg),
    }
    print(f"Filtering + slimming -> {out}  (several minutes for ~6 GB) ...")
    n = copy_to_parquet(con, slim_query(cfg["data"].get("max_rows")), params, out)
    print(f"Done: {n:,} battles, {out.stat().st_size / 1e6:.1f} MB")


def download_per_file(cfg: dict) -> None:
    """Fallback: download one raw file at a time, slim it, delete it.

    At most one raw file (~0.5 GB) is on disk at once. Stops early once max_rows is reached.
    """
    out = slim_file(cfg)
    max_rows = cfg["data"].get("max_rows")
    paths = list_files(cfg)
    if not paths:
        raise SystemExit("No files match formatid; --per-file needs the per-format files.")

    con = connect()
    params = {"formatid": cfg["data"]["formatid"], "min_time": min_upload_time(cfg)}
    kept = 0
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir, parts_dir = Path(tmp, "raw"), Path(tmp, "parts")
        parts_dir.mkdir()
        bar = tqdm(paths, unit="file")
        for i, path in enumerate(bar):
            local = hf_hub_download(
                cfg["data"]["hf_dataset"], path, repo_type="dataset",
                local_dir=raw_dir, token=get_token(),
            )
            params["src"] = [glob_escape(Path(local).as_posix())]
            limit = max_rows - kept if max_rows else None
            kept += copy_to_parquet(con, slim_query(limit), params, parts_dir / f"part{i:03d}.parquet")
            os.remove(local)
            bar.set_postfix(kept=f"{kept:,}")
            if max_rows and kept >= max_rows:
                break
        shutil.rmtree(raw_dir, ignore_errors=True)  # hf_hub_download's small metadata files
        n = copy_to_parquet(
            con, "SELECT * FROM read_parquet($src)",
            {"src": glob_escape(parts_dir.as_posix()) + "/*.parquet"}, out,
        )
    print(f"Done: {n:,} battles, {out.stat().st_size / 1e6:.1f} MB -> {out}")


def inspect(cfg: dict, n: int = 3) -> None:
    """Print a few slimmed logs and the three checks from data/README.md."""
    if not slim_file(cfg).exists():
        raise SystemExit(f"{slim_file(cfg)} doesn't exist yet. Run `python -m src.data.download` first.")
    f = glob_escape(slim_file(cfg).as_posix())
    con = duckdb.connect()
    for battle_id, log_slim in con.execute(
        "SELECT battle_id, log_slim FROM read_parquet($f) LIMIT $n", {"f": f, "n": n}
    ).fetchall():
        print(f"--- {battle_id}\n{log_slim}\n")

    # Patterns are per line (?m). A rating is the 4th field of |player|p1|name|avatar|rating.
    total, six_each, has_win, both_rated = con.execute(
        r"""
        SELECT count(*),
               avg((len(regexp_extract_all(log_slim, '(?m)^\|poke\|p1\|')) = 6
                    AND len(regexp_extract_all(log_slim, '(?m)^\|poke\|p2\|')) = 6)::INT),
               avg(regexp_matches(log_slim, '(?m)^\|win\|')::INT),
               avg((regexp_matches(log_slim, '(?m)^\|player\|p1\|[^|\n]*\|[^|\n]*\|\d+')
                    AND regexp_matches(log_slim, '(?m)^\|player\|p2\|[^|\n]*\|[^|\n]*\|\d+'))::INT)
        FROM read_parquet($f)
        """,
        {"f": f},
    ).fetchone()
    print(f"{total:,} battles in {slim_file(cfg)}")
    print(f"  exactly 6 |poke| lines for p1 and p2:  {100 * six_each:5.1f}%")
    print(f"  has a |win| line:                      {100 * has_win:5.1f}%")
    print(f"  both |player| lines have a rating:     {100 * both_rated:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="list files and count rows per formatid")
    mode.add_argument("--inspect", action="store_true", help="examples + checks on the output file")
    mode.add_argument("--per-file", action="store_true", help="fallback: one raw file at a time")
    args = parser.parse_args()

    cfg = load_config()
    (ROOT / cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    if args.check:
        paths = list_files(cfg, verbose=True)
        check_formats(connect(), hf_sources(cfg, paths), min_upload_time(cfg))
    elif args.inspect:
        inspect(cfg)
    elif args.per_file:
        download_per_file(cfg)
    else:
        download(cfg)
