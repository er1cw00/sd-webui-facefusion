import argparse

def preload(parser: argparse.ArgumentParser):
    print('facefusion preload')
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