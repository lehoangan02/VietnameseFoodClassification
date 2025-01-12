
import argparse
import NeuralNet as lha
from NeuralNet import model
import torch
import os
from torch import nn

num_cpu = os.cpu_count()
print(f"Number of CPU: {num_cpu}")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, help="'train' for training, 'eval' for evaluation", choices=['train', 'eval'])
    parser.add_argument("--path", type=str, help="path to data")
    parser.add_argument("--epoch", type=int, help="number of training iteration")
    parser.add_argument("--model", type=str, help="path to model")
    parser.add_argument("--result_path", type=str, help="path to training/evaluation result")
    parser.add_argument("--id", type=int, help="ID for reference")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_arg()
    if args.result_path is None:
        print("No result path specified")
        quit()
    result_path = args.result_path
    if os.path.exists(result_path):
        print("Using existing result path")
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
        label_path = "TrainLabels.csv"
        TrainDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        TrainDataLoader = lha.DataLoader(TrainDataset, batch_size=16, shuffle=True, num_workers=num_cpu)
        model = torch.load(args.model, weights_only=False)
        model = model.to(lha.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        epoch = args.epoch
        for i in range(epoch):
            print(f"Epoch {i+1}\n-------------------------------")
            lha.train(TrainDataLoader, model, loss_fn, optimizer)
        print("Training done!")
        result_file_name = f"model_{args.id}.pth"
        torch.save(model, result_file_name)

    if args.phase == "eval":
        if args.model is None:
            print("No model specified")
            quit()
        path = args.path
        image_path = path
        label_path = "TrainLabels.csv"
        TestDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        TestDataLoader = lha.DataLoader(TestDataset, batch_size=16, num_workers=num_cpu)
        model.load_state_dict(torch.load('./weights/model5.pth', weights_only=True))
        model = model.to(lha.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        correct, test_loss = lha.test(TestDataLoader, model, loss_fn)
        print("Evaluation done!")
        if args.id is not None:
            id = args.id
        elif args.id is None:
            id = 25
        result_file_name = f"result_{id}.txt"
        with open(result_file_name, "w") as f:
            f.write(f"ID: {id}\n")
            f.write(f"Accuracy: {correct}\n")
            f.write(f"Loss: {test_loss}\n")
        
        
    