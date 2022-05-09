import sys
import argparse
from pathlib import Path
import zipfile
import platform
if platform.system() == 'Windows':
    DEFAULT_OUTPUT = 'c2c7demo-windows.zip'
    BIN_EXT = '.exe'
else:
    DEFAULT_OUTPUT = 'c2c7demo-macos.zip'
    BIN_EXT = ''


def main():
    parser = argparse.ArgumentParser(description='Zip binaries and weights.')
    parser.add_argument('input',
                        help='Input directory: %(default)s',
                        nargs='?',
                        default='target/release')
    parser.add_argument('output',
                        help='Output filename: %(default)s',
                        nargs='?',
                        default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bindir = Path(args.input)
    if not bindir.exists():
        print(f'Input directory "{bindir}" not found.', file=sys.stderr)
        return 1

    with zipfile.ZipFile(args.output,
                         'w',
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for filename in ['c2c7angle', 'c2c7batch']:
            path = bindir / (filename + BIN_EXT)
            zf.write(path, arcname=path.name)
        zf.write(bindir / 'c2c7.onnx', arcname='c2c7.onnx')
    print(f'::set-output name=filename::{args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
