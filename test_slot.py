import json
from typing import List
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from dataset import SeqTaggingClsDataset


# self adding
import os
import pandas as pd
from model import SeqTagger
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference(args):

    assert args.model_name is not None and args.infer_epoch is not None, "Please select model name and inference epoch to do inference."

    # load data
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx = json.loads(tag_idx_path.read_text())
    test_data_path = args.test_file
    test_data = json.loads(test_data_path.read_text())
    test_dataset = SeqTaggingClsDataset(test_data, tag2idx, args.max_len, args.glove_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )


    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = SeqTagger(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=9,
        batch_first=args.batch_first,
        bidirectional=args.bidirectional,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.model_name, f'classifier_{str(args.infer_epoch).zfill(2)}.bin'), map_location=device))
    model.eval()
    
    # prediction list
    final_ids = []
    final_tags = []

    # testing loop
    test_bar = tqdm(
                test_dataloader,
                position=0,
                leave=True,
                total=len(test_dataloader)
    )
    for batch in test_bar:
        batch['tokens'] = batch['tokens'].to(device)
    
        tags_logits = model(batch['tokens'])
        pred_slot_ids = torch.argmax(tags_logits, dim=2)  # (B * max_len)
        
        pred_slot_ids = pred_slot_ids.to('cpu')
        pred_slot_ids = [tags[:batch['len'][idx]] for idx, tags in enumerate(pred_slot_ids)]  # get correct length of a sentence
        pred_slot_tags: List[str] = [' '.join(test_dataset.idx2label(tag.item()) for tag in ids) for ids in pred_slot_ids]  # id2label

        final_ids.extend(batch['id'])
        final_tags.extend(pred_slot_tags)



        
    pred_dict = pd.DataFrame({
        'id': final_ids,
        'tags': final_tags
    })
    pred_save_path = args.pred_file
    pred_dict.to_csv(pred_save_path,index=False)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
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
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")

    # data
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="glove word embedding txt path.",
        default="./glove.840B.300d.txt")
    parser.add_argument("--max_len", type=int, default=50)

    # model
    parser.add_argument('--input_size', type=int, help="word embedding dimension", default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_first", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)


    # data loader
    parser.add_argument("--batch_size", type=int, default=128)


    # selected model name and epoch
    parser.add_argument("--model_name", type=str, default='hidden_s_256')
    parser.add_argument("--infer_epoch", type=int, default=21)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    inference(args)