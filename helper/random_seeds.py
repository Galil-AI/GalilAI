''' Too lazy to come up with random integers yourself?
    Use this script instead!
'''
from numpy import random

def random_values(args):
    rng = random.default_rng()
    rints = rng.integers(low=args.lo, high=args.hi, size=args.n)
    print(sorted(rints))
    
    if __name__ != '__main__':
        return rints


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Return a list of random values to be used as training seeds.'
    )
    parser.add_argument('--lo', type=int, required=False, default=0, help='lowest number in the range')
    parser.add_argument('--hi', type=int, required=False, default=99, help='highest number in the range')
    parser.add_argument('--n', type=int, required=False, default=10, help='how many random values you want')

    args = parser.parse_args()

    random_values(args)

