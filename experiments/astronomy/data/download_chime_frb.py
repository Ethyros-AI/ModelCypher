#!/usr/bin/env python3
"""Download CHIME FRB waterfall data from the Canadian Astronomy Data Center.

Data source: https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data

Usage:
    poetry run python experiments/astronomy/data/download_chime_frb.py --limit 20
    poetry run python experiments/astronomy/data/download_chime_frb.py --list-available
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


# Base URL for CHIME FRB waterfall data
CADC_BASE_URL = (
    "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/"
    "AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data"
)

# Known FRB waterfall files from the CHIME/FRB Catalog 1
# Source: https://www.chime-frb.ca/catalog
# These are a subset of the 600+ FRBs with public waterfall data
KNOWN_FRBS = [
    "FRB20180725A",
    "FRB20180729A",
    "FRB20180814A",
    "FRB20180906A",
    "FRB20180907A",
    "FRB20180908A",
    "FRB20180910A",
    "FRB20180916A",
    "FRB20180917A",
    "FRB20180918A",
    "FRB20180919A",
    "FRB20180923A",
    "FRB20180924A",
    "FRB20180925A",
    "FRB20180927A",
    "FRB20180928A",
    "FRB20181017A",
    "FRB20181019A",
    "FRB20181022A",
    "FRB20181028A",
    "FRB20181030A",
    "FRB20181104A",
    "FRB20181112A",
    "FRB20181117A",
    "FRB20181118A",
    "FRB20181119A",
    "FRB20181123A",
    "FRB20181128A",
    "FRB20181130A",
    "FRB20181201A",
    "FRB20181214A",
    "FRB20181217A",
    "FRB20181220A",
    "FRB20181222A",
    "FRB20181224A",
    "FRB20181226A",
    "FRB20181228A",
    "FRB20181230A",
    "FRB20190102A",
    "FRB20190103A",
    "FRB20190104A",
    "FRB20190106A",
    "FRB20190107A",
    "FRB20190110A",
    "FRB20190111A",
    "FRB20190113A",
    "FRB20190116A",
    "FRB20190117A",
    "FRB20190118A",
    "FRB20190119A",
]


def get_output_dir() -> Path:
    """Get the output directory for downloaded files."""
    output_dir = Path(__file__).parent / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def download_frb(frb_name: str, output_dir: Path, force: bool = False) -> Path | None:
    """Download a single FRB waterfall file.

    Args:
        frb_name: FRB identifier (e.g., "FRB20180725A")
        output_dir: Directory to save the file
        force: Re-download even if file exists

    Returns:
        Path to downloaded file, or None if failed
    """
    filename = f"{frb_name}_waterfall.h5"
    output_path = output_dir / filename
    url = f"{CADC_BASE_URL}/{filename}"

    if output_path.exists() and not force:
        print(f"  [skip] {filename} already exists")
        return output_path

    print(f"  [download] {filename}...")

    try:
        # Download with progress reporting
        def reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                sys.stdout.write(f"\r    Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, reporthook)
        print()  # Newline after progress
        return output_path

    except urllib.error.HTTPError as e:
        print(f"\n  [error] HTTP {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"\n  [error] URL error: {e.reason}")
        return None
    except Exception as e:
        print(f"\n  [error] {e}")
        return None


def list_downloaded(output_dir: Path) -> list[str]:
    """List already downloaded FRB files."""
    files = list(output_dir.glob("FRB*_waterfall.h5"))
    return [f.stem.replace("_waterfall", "") for f in files]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CHIME FRB waterfall data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of FRBs to download (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List available FRB names and exit",
    )
    parser.add_argument(
        "--list-downloaded",
        action="store_true",
        help="List already downloaded FRBs and exit",
    )
    parser.add_argument(
        "--frbs",
        nargs="+",
        help="Specific FRB names to download (e.g., FRB20180725A FRB20180814A)",
    )

    args = parser.parse_args()
    output_dir = get_output_dir()

    if args.list_available:
        print("Available FRBs for download:")
        for frb in KNOWN_FRBS:
            print(f"  {frb}")
        print(f"\nTotal: {len(KNOWN_FRBS)} FRBs")
        return

    if args.list_downloaded:
        downloaded = list_downloaded(output_dir)
        print(f"Downloaded FRBs in {output_dir}:")
        for frb in downloaded:
            print(f"  {frb}")
        print(f"\nTotal: {len(downloaded)} FRBs")
        return

    # Determine which FRBs to download
    if args.frbs:
        frbs_to_download = args.frbs
    else:
        frbs_to_download = KNOWN_FRBS[: args.limit]

    print(f"Downloading {len(frbs_to_download)} FRBs to {output_dir}")
    print()

    success_count = 0
    for frb_name in frbs_to_download:
        result = download_frb(frb_name, output_dir, force=args.force)
        if result is not None:
            success_count += 1

    print()
    print(f"Downloaded {success_count}/{len(frbs_to_download)} FRBs successfully")

    if success_count > 0:
        print(f"\nFiles saved to: {output_dir}")
        print("\nNext steps:")
        print("  poetry run python experiments/astronomy/exp1_frb_intrinsic_dimension.py")


if __name__ == "__main__":
    main()
