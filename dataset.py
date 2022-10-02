from typing import List, Dict

from torch.utils.data import Dataset


from utils import Vocab

# self import package
import torch
import json
import pickle
import math
import numpy as np
from pathlib import Path
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def get_embedding_dic(glove_path: Path, embedding_dim: int) -> Dict:
    dic = dict()
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                values = line.split()
                if not len(values) == embedding_dim+1:
                    continue
                else:
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    dic[word] = vector
            except ValueError:
                pass
    
    return dic

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        glove_path: Path,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.glove_path = glove_path
        
        self.embeddings_dict = get_embedding_dic(self.glove_path, embedding_dim=300)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)


    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        tokenizer = get_tokenizer("basic_english")
        tensors = []
        intents = [sample['intent'] for sample in samples]
        for sample in samples:
            tokens = tokenizer(sample['text']) 
            intent = sample['intent']
            for token in tokens:  # remove word not in embedding_dict
                if not token in self.embeddings_dict.keys():
                    tokens.remove(token)
            tensor = torch.stack([torch.from_numpy(self.embeddings_dict[token]) for token in tokens]) # shape: (number of token x embedding_dim)
            if tensor.shape[0] >= self.max_len:
                tensor = tensor[:self.max_len]
            else:
                tensor = torch.cat([tensor, torch.zeros(self.max_len-tensor.shape[0], 300)])
            tensors.append(tensor)
        batch_word_tensors = torch.stack([tensor for tensor in tensors])  #　shape: (B x max_length(or the length of the longest text) x embedding_dim)

        return{
            'text': batch_word_tensors,
            'intent': intents,
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn

        raise NotImplementedError

if __name__ == '__main__':

    with open('data\\intent\\train.json') as f:
        data = json.load(f)

    with open('cache\\intent\\vocab.pkl', 'rb') as f:
        voc = pickle.load(f)  #　<class 'utils.Vocab'>

    with open('cache\\intent\\intent2idx.json') as f:
        label_mapping = json.load(f)

    dataset = SeqClsDataset(
        data=data,
        vocab=voc,
        label_mapping=label_mapping,
        max_len= 50,
        glove_path= 'glove.840B.300d.txt'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    for i, batch in enumerate(dataloader):
        print(batch['text'].shape)

        if i == 100:
            break



