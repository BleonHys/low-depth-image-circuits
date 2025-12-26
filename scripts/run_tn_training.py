#!/usr/bin/env python3
import argparse
import os

from classifier.utils.tensor_network_training import main


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def run(args) -> None:
    os.makedirs(args.basepath, exist_ok=True)
    config = {
        "model_name": args.model,
        "basepath": args.basepath,
        "data_dir": args.data_dir,
        "patched": args.patched,
        "dataset_name": args.dataset,
        "n_factors": args.n_factors,
        "warmstart": args.warmstart,
        "compression_depth": args.compression_depth,
        "fold": args.fold,
        "chi_final": args.chi_final,
        "n_samples_warm_start": args.n_samples_warm_start,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    main(config, use_ray=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tensor-network training with a simple CLI.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["mps", "mpo"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--basepath", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--patched", type=str_to_bool, default=False)
    parser.add_argument("--n_factors", type=int, default=1)
    parser.add_argument("--warmstart", type=str_to_bool, default=False)
    parser.add_argument("--compression_depth", type=int, default=0)
    parser.add_argument("--n_samples_warm_start", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chi_final", type=int, default=16)

    run(parser.parse_args())
