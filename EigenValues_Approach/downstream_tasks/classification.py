import warnings

warnings.filterwarnings("ignore")

import torch
import random
import argparse
import numpy as np
import scanpy as sc
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import TensorDataset, DataLoader


class Classifier(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, class_num),
        )

    def forward(self, x):
        return self.classifier(x)


def main(args):
    original_adata = sc.read_h5ad(args.original_path)
    sc.pp.normalize_total(original_adata, target_sum=1e4)
    sc.pp.log1p(original_adata)
    sc.pp.scale(original_adata, max_value=10)
    original_label = original_adata.obs[args.type_key].cat.codes
    label_map = {
        original_adata.obs[args.type_key].cat.categories[i]: i
        for i in range(original_label.max() + 1)
    }
    test_data = torch.from_numpy(original_adata.X).float()
    test_label = torch.from_numpy(original_label.values).long()

    metacell_adata = sc.read_h5ad(args.metacell_path)
    sc.pp.scale(metacell_adata, max_value=10)
    train_data = torch.from_numpy(metacell_adata.X).float()
    train_label = [label_map[i] for i in metacell_adata.obs[args.type_key]]
    train_label = torch.tensor(train_label).long()

    dataset_train = TensorDataset(train_data, train_label)
    print("Training metacell number", len(dataset_train))
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    dataset_test = TensorDataset(test_data, test_label)
    print("Test single cell number", len(dataset_test))
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size * 4, shuffle=False, drop_last=False
    )

    device = torch.device(args.device)

    print("======= Training Start =======")
    classifier = Classifier(train_data.size(1), train_label.max() + 1).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    classifier.train()
    for epoch in range(args.train_epoch):
        loss_epoch = 0
        for i, (data, label) in enumerate(dataloader_train):
            data = data.to(device)
            label = label.to(device)

            pred = classifier(data)
            loss = nn.functional.cross_entropy(pred, label)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, loss_epoch / (i + 1)))
    print("======= Training End =======")

    print("")

    print("======= Testing Start =======")
    classifier.eval()
    preds, labels = [], []
    for i, (data, label) in enumerate(dataloader_test):
        data = data.to(device)
        pred = classifier(data).detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label.numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    acc = (preds == labels).mean()
    print("* Classification Accuracy: {}".format(acc))

    print("Per Class Accuracy:")
    label_map_inv = {v: k for k, v in label_map.items()}
    for i in range(test_label.max() + 1):
        print(label_map_inv[i], end=": ")
        print((preds[labels == i] == i).mean())

    balanced_acc = balanced_accuracy_score(labels, preds)
    print("* Balanced Accuracy: {}".format(balanced_acc))
    print("======= Testing End =======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data configs
    parser.add_argument("--original_path", type=str)
    parser.add_argument("--metacell_path", type=str)
    parser.add_argument("--type_key", type=str, default="celltype")

    # Training configs
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_epoch", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Randomization
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)
