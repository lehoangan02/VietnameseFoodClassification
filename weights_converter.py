import torch
def CudaToMps(weights_path):
    weights = torch.load(weights_path, map_location=torch.device('mps'), weights_only=True)
    new_weights_path = weights_path.replace(".pth", "_mps.pth")
    torch.save(weights, new_weights_path)
    print(f"Converted weights saved to {new_weights_path}")
    return new_weights_path
def MpsToCuda(weights_path):
    weights = torch.load(weights_path, map_location=torch.device('cuda'), weights_only=True)
    new_weights_path = weights_path.replace(".pth", "_cuda.pth")
    torch.save(weights, new_weights_path)
    print(f"Converted weights saved to {new_weights_path}")
    return new_weights_path
def CudaToCpu(weights_path):
    weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    new_weights_path = weights_path.replace(".pth", "_cpu.pth")
    torch.save(weights, new_weights_path)
    print(f"Converted weights saved to {new_weights_path}")
    return new_weights_path
def CpuToCuda(weights_path):
    weights = torch.load(weights_path, map_location=torch.device('cuda'), weights_only=True)
    new_weights_path = weights_path.replace(".pth", "_cuda.pth")
    torch.save(weights, new_weights_path)
    print(f"Converted weights saved to {new_weights_path}")
    return new_weights_path

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to weights")
    parser.add_argument("--source", type=str, help="source device", choices=['cuda', 'mps', 'cpu'])
    parser.add_argument("--target", type=str, help="target device", choices=['cuda', 'mps', 'cpu'])
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_arg()
    if args.path is None:
        print("No weights path specified")
        quit()
    if args.source == "cuda" and args.target == "mps":
        CudaToMps(args.path)
    elif args.source == "mps" and args.target == "cuda":
        MpsToCuda(args.path)
    elif args.source == "cuda" and args.target == "cpu":
        CudaToCpu(args.path)
    elif args.source == "cpu" and args.target == "cuda":
        CpuToCuda(args.path)
    else:
        print("Invalid source/target device")
        print("Mps and Cpu are not supported")
        quit()