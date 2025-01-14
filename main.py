
import argparse
import NeuralNet as lha
import torch
import os
from torch import nn

num_cpu = os.cpu_count()
print(f"Number of CPU: {num_cpu}")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, help="'train' for training, 'eval' for evaluation", choices=['train', 'eval', 'test'])
    parser.add_argument("--path", type=str, help="path to data")
    parser.add_argument("--epoch", type=int, help="number of training iteration")
    parser.add_argument("--weight", type=str, help="path to weight")
    parser.add_argument("--result_path", type=str, help="path to training/evaluation result")
    parser.add_argument("--id", type=int, help="ID for reference")
    parser.add_argument("--model", type=str, help="model name", choices=['VNFNeuNet', 'VGG16', 'DenseNet', 'ResNet'])
    parser.add_argument("--resume_train", action='store_true', help="resume training")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_arg()
    if args.result_path is None:
        print("No result path specified")
        quit()
    result_path = args.result_path
    if args.model is None:
        print("No model specified")
        quit()
    if os.path.exists(result_path):
        print("Using existing result path")
    if args.id is not None:
        id = args.id
    elif args.id is None:
        id = 9999
    if args.phase == "train":

        # load data
        if args.epoch is None:
            print("Epoch must be specified for train")
            quit()
        elif args.model is None:
            print("No model specified")
            quit()
        path = args.path
        # image_path = path + "/images" this will be updated along with the dataset
        # label_path = path + "/labels" this will be updated along with the dataset
        image_path = os.path.join(path, 'images')
        label_path = os.path.join(path, "label/labels.csv")
        if args.weight is None:
            print("No weight specified")
        else:
            weights_path = './weights/' + args.weight
        TrainDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        TrainDataLoader = lha.DataLoader(TrainDataset, batch_size=16, shuffle=True, num_workers=num_cpu)

        # set up model
        model, loss_fn, optimizer = lha.NeuralNetFactory().create(args.model)
        if (args.resume_train):
            model.load_state_dict(torch.load(weights_path, weights_only=True))
        
        # training
        epoch = args.epoch
        for i in range(epoch):
            print(f"Epoch {i+1}\n-------------------------------")
            lha.train(TrainDataLoader, model, loss_fn, optimizer)
        print("Training done!")
        result_file_name = f"model_{args.model}_{args.id}.pth"
        torch.save(model.state_dict(), result_file_name)
    if args.phase == "eval":
        print("Evaluation phase")
        # load data
        if args.model is None:
            print("No model specified")
            quit()
        path = args.path
        image_path = os.path.join(path, "images")
        label_path = os.path.join(path, "label/labels.csv")
        weights_path = './weights/' + args.weight
        print(f"Image path: {image_path}")
        TestDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        TestDataLoader = lha.DataLoader(TestDataset, batch_size=16, num_workers=num_cpu)

        # set up model
        model, loss_fn, optimizer = lha.NeuralNetFactory().create(args.model)
        if (args.resume_train):
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            # model = model.to(lha.device)

        # evaluation
        correct, test_loss = lha.eval(TestDataLoader, model, loss_fn)
        print("Evaluation done!")

        # save result
        result_file_name = f"result_{id}.txt"
        result_file_name = os.path.join(result_path, result_file_name)
        print(f"Result file: {result_file_name}")
        with open(result_file_name, "w") as f:
            f.write(f"ID: {id}\n")
            f.write(f"Path: {path}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Weights: {weights_path}\n")
            f.write(f"Accuracy: {correct}\n")
            f.write(f"Loss: {test_loss}\n")
        print("Result saved!")
    if args.phase == 'test':
        print("Test phase")
        # load data
        if args.model is None:
            print("No model specified")
            quit()
        path = args.path
        image_path = os.path.join(path, "images")
        label_path = os.path.join(path, "label/labels.csv")
        weights_path = './weights/' + args.weight
        print(f"Image path: {image_path}")
        TestDataset = lha.fd.VietnameseFoodDataset(label_path, image_path)
        TestDataLoader = lha.DataLoader(TestDataset, batch_size=16, num_workers=num_cpu)

        # set up model
        model, loss_fn, optimizer = lha.NeuralNetFactory().create(args.model)
        if (args.resume_train):
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            # model = model.to(lha.device)
        else:
            pass
        # test
        lha.test(TestDataLoader, model, loss_fn)
        print("Test done!")
        print("Result saved!")
        
    