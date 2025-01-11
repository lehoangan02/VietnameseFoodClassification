import food_dataset as fd
from torch.utils.data import DataLoader
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("echo", help="echo the string you use here")
# args = parser.parse_args()
# print(args.echo)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbosity", help="increase output verbosity")
args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")

# TrainDataset = fd.VietnameseFoodDataset("TrainLabels.csv", "./Train")

# TrainDataLoader = DataLoader(TrainDataset, batch_size=64, shuffle=True)

# TrainFeatures, TrainLabels = next(iter(TrainDataLoader))
# print(f"Feature batch shape: {TrainFeatures.shape}")
# print(f"Label batch shape: {TrainLabels.shape}")
