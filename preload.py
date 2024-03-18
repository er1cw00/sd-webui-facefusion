import argparse

def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--facefusion-skip-download",
        action="store_true",
        help="omit automate downloads and lookups",
    )
    parser.add_argument(
        "--facefusion-proxy",
        type=str,
        default=None,
        help="facefusion model download proxy",
    )
    parser.add_argument(
        "--facefusion-disable-install",
        action="store_false",
        help="facefusion not install dependency",
    )