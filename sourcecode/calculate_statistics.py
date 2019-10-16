import argparse, os
import numpy as np
from MSG_GAN.FID.inception import InceptionV3
from MSG_GAN.FID.fid_score import _compute_statistics_of_path


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_path", action="store", type=str,
                        default=None,
                        help="path to the real images/ images for which fid is to be calculated")

    parser.add_argument("--save_path", action="store", type=str,
                        default=None,
                        help="path for saving the calculated fid statistics")

    args = parser.parse_args()

    return args


def main(args, myargs=None):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model.cuda()

    m, s = _compute_statistics_of_path(args.images_path, model, 64, 2048, True)

    np.savez(args.save_path, mu=m, sigma=s)


def run(args1, myargs):
  myargs.config = getattr(myargs.config, args1.command)
  myargs.args = args1
  args = parse_arguments()
  for k, v in myargs.config.items():
    setattr(args, k, v)
  args.images_path = os.path.expanduser(args.images_path)
  args.save_path = os.path.expanduser(args.save_path)
  os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
  main(args, myargs)
  pass


if __name__ == '__main__':
    main(parse_arguments())
