import datetime
import json
import pathlib
import re
import tqdm

from .. import datafiles

FILENAME_RE = re.compile("^(\d{8})-([^-]*)\.records$")

VERSION = 3


def records_to_parquet(
    in_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    averaging: str,
    begin_date: datetime.date = None,
    end_date: datetime.date = None,
    make_out_dir: bool = True,
):
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)

    metadata_path = pathlib.Path(out_dir) / "metadata.json"

    if not out_dir.exists():
        if make_out_dir:
            out_dir.mkdir(parents=True)
        else:
            raise FileNotFoundError(f"Output folder not found: {out_dir!r}")

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("version", None) != VERSION:
            metadata = None
    except FileNotFoundError:
        metadata = None

    if metadata is None:
        metadata = {"version": VERSION, "files": {}}

    files_metadata = metadata["files"]

    files = sorted(list(pathlib.Path(in_dir).iterdir()))

    updated_files = []

    for f in files:
        match = FILENAME_RE.match(f.name)
        if not match or match[2] != averaging:
            continue
        datestr = match[1]
        date = datetime.date(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8]))
        if begin_date and date < begin_date:
            continue
        if end_date and date >= end_date:
            continue

        update = False
        if f.name in files_metadata:
            stat = f.stat()
            size = stat.st_size
            fmd = files_metadata[f.name]
            if fmd.get("size", -1) != size:
                update = True
        else:
            update = True

        outfile = out_dir / f"{f.name}.parquet"

        if not outfile.exists():
            update = True

        if update:
            updated_files.append((f, outfile))

    if updated_files:
        for f, outfile in tqdm.tqdm(updated_files):
            r = datafiles.RecordsFiles()
            r.load(f)
            df = r.to_polars(tz="UTC")
            df.write_parquet(outfile, statistics=True)
            files_metadata[f.name] = {"size": f.stat().st_size}

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
