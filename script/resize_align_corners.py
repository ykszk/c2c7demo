import sys
import argparse
from logging import basicConfig, getLogger, INFO

basicConfig(level=INFO, format='%(asctime)s %(levelname)s :%(message)s')
logger = getLogger(__name__)

import onnx


def main():
    parser = argparse.ArgumentParser(
        description='Change resize mode in the ONNX model to align_corners.')
    parser.add_argument('input',
                        help='Input onnx filename:',
                        metavar='<input>')
    parser.add_argument('output',
                        help='Output onnx filename',
                        metavar='<output>')

    args = parser.parse_args()

    logger.info("Loading Model %s", args.input)
    model = onnx.load_model(args.input)
    for node in model.graph.node:
        if (node.op_type == "Resize"
                and node.attribute[0].s == b'pytorch_half_pixel'):
            logger.info("Update %s", node.name)
            node.attribute[0].s = b'align_corners'
    logger.info("Save output %s", args.output)
    onnx.save_model(model, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
