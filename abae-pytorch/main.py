# -*- coding: utf-8 -*-
import torch
from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--word-vectors-path", "-wv",
                        dest="wv_path", type=str, metavar='<str>',
                        default="/home/uyplayer/Github/ABAE/abae-pytorch/word_vectors/Electronics_5.json.txt.w2v",
                        help="path to word vectors file")

    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=50,
                        help="Batch size for training")

    parser.add_argument("--aspects-number", "-as", dest="aspects_number", type=int, default=40,
                        help="A total number of aspects")

    parser.add_argument("--ortho-reg", "-orth", dest="ortho_reg", type=float, default=0.1,
                        help="Ortho-regularization impact coefficient")

    parser.add_argument("--epochs", "-e", dest="epochs", type=int, default=1,
                        help="Epochs count")

    parser.add_argument("--optimizer", "-opt", dest="optimizer", type=str, default="adam", help="Optimizer",
                        choices=["adam", "adagrad", "sgd"])

    parser.add_argument("--negative-samples", "-ns", dest="neg_samples", type=int, default=5,
                        help="Negative samples per positive one")

    parser.add_argument("--dataset-path", "-d", dest="dataset_path", type=str, default="/home/uyplayer/Github/ABAE/abae-pytorch/data/Electronics_5.json.txt",
                        help="Path to a training texts file. One sentence per line, tokens separated wiht spaces.")

    parser.add_argument("--maxlen", "-l", type=int, default=201,
                        help="Max length of the considered sentence; the rest is clipped if longer")

    args = parser.parse_args()

    w2v_model = get_w2v(args.wv_path)  # 加载word2vec模型
    wv_dim = w2v_model.vector_size  # 获取长度
    y = torch.zeros(args.batch_size, 1) # 设一个矩阵（batch_size,1） 就要是y来做分类标签

    # 模型
    model = ABAE(wv_dim=wv_dim,
                 asp_count=args.aspects_number,
                 init_aspects_matrix=get_centroids(w2v_model, aspects_count=args.aspects_number))
    print(model)

    criterion = torch.nn.MSELoss(reduction="sum")

    optimizer = None
    scheduler = None

    # if args.optimizer == "cycsgd":
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.05, mode="triangular2")
    # elif args.optimizer == "adam":

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
    else:
        raise Exception("Optimizer '%s' is not supported" % args.optimizer)

    for t in range(args.epochs):

        print("Epoch %d/%d" % (t + 1, args.epochs))

        data_iterator = read_data_tensors(args.dataset_path, args.wv_path,
                                          batch_size=args.batch_size, maxlen=args.maxlen)

        for item_number, (x, texts) in enumerate(data_iterator):

            x = torch.from_numpy(x)

            # extracting bad samples from the very same batch; not sure if this is OK, so todo
            negative_samples = torch.stack(
                tuple([x[torch.randperm(x.shape[0])[:args.neg_samples]] for _ in range(args.batch_size)]))

            # prediction
            y_pred = model(x, negative_samples)

            # error computation
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(epoch=t)

            if item_number % 1000 == 0:

                print(item_number, "batches, and LR:", optimizer.param_groups[0]['lr'])

                for i, aspect in enumerate(model.get_aspect_words(w2v_model)):
                    print(i + 1, " ".join(["%10s" % a for a in aspect]))

                print("Loss:", loss.item())
                print()
