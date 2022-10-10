import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from dataset import SeqClsDataset


# self adding
import os
import pandas as pd
import math
from model import SeqClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference(args):
    # load data
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx = json.loads(intent_idx_path.read_text())
    test_data_path = args.data_dir / "test.json"
    test_data = json.loads(test_data_path.read_text())
    test_dataset = SeqClsDataset(test_data, intent2idx, args.max_len, args.glove_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )


    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = SeqClassifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=150,
        batch_first=args.batch_first,
        bidirectional=args.bidirectional,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.model_name, f'classifier_{str(args.infer_epoch).zfill(2)}.bin')))
    model.eval()
    
    # prediction list
    ids = []
    intents = []

    # testing loop
    test_bar = tqdm(
                test_dataloader,
                position=0,
                leave=True,
                total=len(test_dataloader)
    )
    for batch in test_bar:
        batch['text'] = batch['text'].to(device)

        intent_logits = model(batch['text'])
        pred_intents_id = torch.argmax(intent_logits, dim=1)  # B
        
        pred_intents_id = pred_intents_id.to('cpu')

        ids.extend(batch['id'])
        intents.extend([test_dataset.idx2label(id.item()) for id in pred_intents_id])
        
    pred_dict = pd.DataFrame({
        'id': ids,
        'intent': intents
    })
    pred_save_path = os.path.join(args.ckpt_dir, args.model_name, 'intent_pred.csv')
    pred_dict.to_csv(pred_save_path,index=False)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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


    # data loader
    parser.add_argument("--batch_size", type=int, default=128)


    # selected model name and epoch
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--infer_epoch", type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    inference(args)