import argparse
import time

from preprocess import DataProcessor
from train import Trainer

from torchtext import data
from torch.backends import cudnn
from torch import cuda


def main(args):
    if cuda.is_available():
        cuda.set_device(int(args.gpu_num))
        cudnn.benchmark = True

    start_time = time.time()

    dp = DataProcessor(args.src_lang, args.trg_lang)
    train_dataset, valid_dataset, test_dataset, vocabs = dp.load_data('fra.txt', 'english_french.pkl')

    print("Elapsed Time: %1.3f \n" % (time.time() - start_time))

    print("=========== Data Stat ===========")
    print("Train: ", len(train_dataset))
    print("Valid: ", len(valid_dataset))
    print("Test: ", len(test_dataset))
    print("=================================")

    train_loader = data.BucketIterator(dataset=train_dataset, batch_size=args.batch_size,
                                       repeat=False, shuffle=True, sort_within_batch=True,
                                       sort_key=lambda x: len(x.src))

    val_loader = data.BucketIterator(dataset=valid_dataset, batch_size=args.batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True,
                                     sort_key=lambda x: len(x.src))

    trainer = Trainer(train_loader, val_loader, vocabs, args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Language setting
    parser.add_argument('--dataset', type=str, default='europarl')
    parser.add_argument('--src_lang', type=str, default='fr')
    parser.add_argument('--trg_lang', type=str, default='en')
    parser.add_argument('--max_len', type=int, default=50)

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=2)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # Training setting
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_epoch', type=int, default=100)

    # Path
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--train_path', type=str, default='./data/training/europarl-v7.fr-en')
    parser.add_argument('--val_path', type=str, default='./data/dev/newstest2013')

    # Dir.
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--sample', type=str, default='sample')

    # Misc.
    parser.add_argument('--gpu_num', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
