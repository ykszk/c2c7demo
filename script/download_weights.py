import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Download trained weigts from google drive.')
    parser.add_argument('dst', help='Output directory: %(default)s', nargs='?', default='target/release')
    args = parser.parse_args()


    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    gdid, filename = '1Evc5e17_klcBSNw-8-fnTREYV1LZsOlM', 'c2c7.onnx'
    output = dst / filename
    import gdown
    gdown.download(id=gdid, output=str(output))


if __name__ == '__main__':
    sys.exit(main())