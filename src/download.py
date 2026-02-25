import argparse
import os
import sys

COMPONENTS = ["images", "labels"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download KITTI dataset")

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory"
    )

    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        default=COMPONENTS,
        help="Components to download"
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction"
    )

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        # Create directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Simulate download
        for comp in args.components:
            print(f"Downloading {comp}...")

        # Optional extraction
        if args.no_extract:
            print("Skipping extraction.")
        else:
            print("Extracting files...")

        print("Done.")
        return 0

    except Exception as e:
        print("Error:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())