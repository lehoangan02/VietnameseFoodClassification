
import argparse
import NeuralNet as lha
import torch

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str, help="'train' for training, 'eval' for evaluation", choices=['train', 'eval'])
    parser.add_argument("path", type=str, help="path to data")
    parser.add_argument("--epoch", type=int, help="number of training iteration")
    parser.add_argument("--model", type=str, help="path to model")
    parser.add_argument("result_path", type=str, help="path to training/evaluation result")
    parser.add_argument("--id", type=int, help="ID for reference")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_arg()
    if args.phase == "train":
        if args.epoch is None:
            print("Epoch must be specified for train")
            quit()
        elif args.model is None:
            print("No model specified")
            quit()
        path = args.path
        # image_path = path + "/images" this will be updated along with the dataset
        # label_path = path + "/labels" this will be updated along with the dataset
        image_path = path
        label_path = "labels.csv"
        TrainDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        model = torch.load(path, weights_only=False)
        model = model.to(lha.device)
        epoch = args.epoch
        for i in range(epoch):
            print(f"Epoch {i+1}\n-------------------------------")
            lha.train(TrainDataset, lha.model, lha.loss_fn, lha.optimizer)
        print("Training done!")
        torch.save(model, args.result_path)

    if args.phase == "eval":
        if args.model is None:
            print("No model specified")
            quit()
        path = args.path
        image_path = path
        label_path = "labels.csv"
        TestDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        model = torch.load(path, weights_only=False)
        model = model.to(lha.device)
        lha.test(TestDataset, model, lha.loss_fn)
        print("Evaluation done!")
        if args.id is not None:
            id = args.id
        elif args.id is None:
            id = 25
        with open(args.result_path, "w") as f:
            f.write(f"ID: {id}\n")
            f.write(f"Accuracy: {lha.correct}\n")
            f.write(f"Loss: {lha.test_loss}\n")
        
        
    