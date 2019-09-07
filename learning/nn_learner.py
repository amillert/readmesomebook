from custom.custom_dataset import BookDataset
from models.test import NNModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time


def learn(args):
    dataset = BookDataset(args)
    print(f"amount of data: {len(dataset)}")
    batches = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    vocab_size = dataset.vocab_size
    FAKE = dataset.FAKE

    criterion = nn.NLLLoss(ignore_index=FAKE)

    model = NNModel(vocab_size, 500, FAKE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.eta)

    num_batches = len(dataset) // args.batch_size
    print(f"number of upcoming batches: {num_batches}")

    for epoch in range(args.epochs):
        tick = time.time()
        loss_total = 0.0
        batch_count = 0
        for target, context in batches:
            batch_count += 1
            loss = criterion(model(context), target)
            loss_total += loss.item()

            if not batch_count % 10 and batch_count > 1:
                print(f"loss of batch {batch_count} during epoch {epoch + 1} is: {loss.item() / batch_count}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("~~~~~~~~~~~~~~~~~~")
        print(f"end of epoch: {epoch + 1} out of {args.epochs}")
        print(f"time per epoch: {time.time() - tick}")
        print(f"mean loss: {loss_total / num_batches}")
        print(f"total loss: {loss_total}")
        print("~~~~~~~~~~~~~~~~~~")

        # weights_in = model.embedding_layer_in.weight.detach().numpy()
        # weights_out = model.embedding_layer_out.weight.detach().numpy()
        # weights = (weights_in + weights_out) / 2
        # words = dataset.vocab

        # save_weights(args.weights_data_path, dataset, weights, words, epoch + 1)

def save_weights(save_weights_path, dataset, weights, words, epoch):
    with open(save_weights_path, "w") as file:
        for w in words:
            vec = weights[dataset.word2idx[w]].reshape(1, -1)[0]
            file.write(f"{w},{' '.join(vec.astype(str).tolist())}\n")

    print("epoch {epoch + 1} generated and saved")

