"""Create the shared seed-10 Fine initialization for controlled ablations."""

import argparse

import torch

from adaptcliplib import BridgeAdaptCLIPV12
from tools import setup_seed


def main(args):
    setup_seed(args.seed)
    model = BridgeAdaptCLIPV12(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
    )
    torch.save({
        'seed': args.seed,
        'architecture': 'BridgeAdaptCLIPV12-strip',
        'purpose': 'shared compatible initialization for DeRCLIP-F ablations',
        'bridgeadaptclip_initial_state': model.state_dict(),
    }, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--fusion_channels', type=int, default=128)
    parser.add_argument('--structural_channels', type=int, default=128)
    parser.add_argument('--strip_kernel', type=int, default=5)
    parser.add_argument('--structural_input_size', type=int, default=1024)
    main(parser.parse_args())
