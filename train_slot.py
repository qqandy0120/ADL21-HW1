import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

# self adding
import os
import pandas as pd
import math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    assert args.model_name is not None, "please key in model name."
    # with open(args.cache_dir / "vocab.pkl", "rb") as f:
    #     vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, tag2idx, args.max_len, args.glove_path)
        for split, split_data in data.items()
    }
    # crecate DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
        for split, dataset in datasets.items()
    }
    
    # init model and move model to target device(cpu / gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SeqTagger(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=9,
        batch_first=args.batch_first,
        bidirectional=args.bidirectional,
    ).to(device)

    # init optimizer
    optimizer = getattr(optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # init criterion
    criterion = getattr(nn, args.criterion)()

    # Train / Evaluation Step
    save_path = os.path.join(args.ckpt_dir, args.model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    best_loss = math.inf
    early_stop_counter = 0
    saved_epoch_lst = []
    final_pred_lst = []
    final_true_lst = []
    for epoch in range(args.num_epoch):
        # Training loop
        train_bar = tqdm(
            dataloaders[TRAIN],
            position=0,
            leave=True,
            total=len(dataloaders[TRAIN])
        )
        train_loss_lst = []
        model.train()
        for t_batch in train_bar:
            t_batch['tokens'] = t_batch['tokens'].to(device)
            t_batch['tags'] = t_batch['tags'].to(device)
            tag_logits = model(t_batch['tokens'])
            loss = criterion(tag_logits, t_batch['tags'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_lst.append(loss.detach().item())

            train_bar.set_description(f'Epoch [{epoch}/{args.num_epoch}]')
            train_bar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(train_loss_lst) / len(train_loss_lst)

        #Evaluation loop
        eval_bar = tqdm(
            dataloaders[DEV],
            position=0,
            leave=True,
            total=len(dataloaders[DEV])
        )
        model.eval()
        val_loss_lst = []
        pred_tags_lst = []
        true_tags_lst = []
        for e_batch in eval_bar:
            e_batch['tokens'] = e_batch['tokens'].to(device)
            e_batch['tags'] = e_batch['tags'].to(device)
            with torch.no_grad():
                tag_logits = model(e_batch['tokens'])

                # seqeval evaluation
                pred_ids = torch.argmax(tag_logits, dim=2)
                pred_ids = pred_ids.to('cpu')
                pred_ids = [tags[:e_batch['len'][idx]] for idx, tags in enumerate(pred_ids)]
                pred_tags: List[List[str]] = [[datasets[DEV].idx2label(tag.item()) for tag in ids] for ids in pred_ids]
                true_ids = torch.argmax(e_batch['tags'], dim=2)
                true_ids = [tags[:e_batch['len'][idx]] for idx, tags in enumerate(true_ids)]
                true_tags: List[List[str]] = [[datasets[DEV].idx2label(tag.item()) for tag in ids] for ids in true_ids]
                pred_tags_lst.extend(pred_tags)
                true_tags_lst.extend(true_tags)

                loss = criterion(tag_logits, e_batch['tags'])
            val_loss_lst.append(loss)
        mean_val_loss = sum(val_loss_lst) / len(val_loss_lst)

        print(f'Epoch[{epoch}/{args.num_epoch}]: Train Loss: {mean_train_loss:.5f}, Valid Loss: {mean_val_loss:.5f}')

        if mean_val_loss < best_loss:
            saved_epoch_lst.append(str(epoch).zfill(2))
            while len(saved_epoch_lst) > 5:
                remove_epoch = saved_epoch_lst.pop(0)
                os.remove(os.path.join(save_path, f'classifier_{remove_epoch}.bin'))
            torch.save(model.state_dict(), os.path.join(save_path, f'classifier_{str(epoch).zfill(2)}.bin'))

            best_loss = mean_val_loss
            early_stop_counter = 0
            print(f'Saving model with loss: {best_loss:5f}')
            final_pred_lst, final_true_lst = pred_tags_lst, true_tags_lst
        else:
            early_stop_counter += 1
        if early_stop_counter >= args.duration:
            print(f'Early Stop at epoch {epoch}')
            break
    with open(os.path.join(save_path, 'report.txt'), 'w') as f:
        f.write(classification_report(final_true_lst, final_pred_lst, mode='strict', scheme=IOB2))

    with open(os.path.join(save_path, 'args.json') , 'w') as f:
        args_dict = vars(args)
        for key, value in args_dict.items():
            args_dict[key] = str(value)
        args_dict['best_val_loss'] = best_loss
        args_dict['stop_at_epoch'] = epoch
        json.dump(args_dict, f)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="glove word embedding txt path.",
        default="./glove.840B.300d.txt")
    parser.add_argument("--max_len", type=int, default=50)

    # model
    parser.add_argument('--input_size', type=int, help="word embedding dimension", default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_first", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # criterion
    parser.add_argument("--criterion", type=str, default='BCEWithLogitsLoss')

    # training
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--duration", type=int, default=30)

    # model save name
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)