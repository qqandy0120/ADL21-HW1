from typing import List, Dict

from torch.utils.data import Dataset


from utils import Vocab

# self import package
import re
import torch
import json
import pickle
import math
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


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

def get_alter_token(token):
    if re.search(r'\d/\d', token):
        return 'date'
    if re.search(r'\d:\d', token):
        return 'time'
    if re.findall(r'\b\w+\b', token):
        return re.findall(r'\b\w+\b', token)[0]
    return token

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        # vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        glove_path: Path,
    ):
        self.data = data
        # self.vocab = vocab
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
        tensors = []
        try:
            intents = [self.label_mapping[sample['intent']] for sample in samples]  # [2, 3, 10, 7]
        except:
            pass  #　test dataset does not have intents
        ids = [sample['id'] for sample in samples]
        for sample in samples:
            tokens = word_tokenize(sample['text']) 
            # remove token not in embedding_dict and mapping token
            tensor = torch.stack([torch.from_numpy(self.embeddings_dict[token]) for token in tokens if token in self.embeddings_dict.keys()]) # shape: (number of token in embeddings_dict x embedding_dim)

            if tensor.shape[0] >= self.max_len:
                tensor = tensor[:self.max_len]
            else:
                tensor = torch.cat([tensor, torch.zeros(self.max_len-tensor.shape[0], 300)])
            tensors.append(tensor)
        batch_word_tensors = torch.stack([tensor for tensor in tensors])  #　shape: (B x max_length x embedding_dim)
        try:
            return{
                'text': batch_word_tensors,
                'intent': torch.Tensor(np.eye(self.num_classes)[intents]),  # convert List[int] to Tensor(one-vector)
                'id': ids,  # List[str]
            }
        except:  # for test dataset
            return{
                'text': batch_word_tensors,
                'id': ids,  # List[str]
            }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch_tokens, batch_tags, batch_ids, batch_lens = [], [], [], []
        is_test = False
        for sample in samples:
            if 'tags' not in sample.keys():
                is_test = True
            tokens_vector = []
            tokens, id = sample['tokens'], sample['id']
            lens = len(tokens)
            for token in tokens:
                alt_token = get_alter_token(token)
                if token in self.embeddings_dict.keys():
                    tokens_vector.append(torch.from_numpy(self.embeddings_dict[token]))
                elif alt_token in self.embeddings_dict.keys():
                    tokens_vector.append(torch.from_numpy(self.embeddings_dict[alt_token]))
                else:
                    random_vector = torch.rand(300)
                    tokens_vector.append(random_vector)

            # fix length of a sentence
            tokens_vector = torch.stack(tokens_vector)
            if tokens_vector.shape[0] >= self.max_len:
                tokens_vector = tokens_vector[:self.max_len]
            else:
                padding_vector = torch.zeros(self.max_len-tokens_vector.shape[0], 300)
                tokens_vector= torch.cat([tokens_vector, padding_vector])
            
            
            if not is_test:
                tags: List[str] = sample['tags']
                tags: List[int] = [self.label_mapping[tag] for tag in tags]
                tags_vector = torch.from_numpy(np.eye(self.num_classes)[tags])  # (num_token * num_label)

                # fix length of tags
                if tags_vector.shape[0] >= self.max_len:
                    tags_vector = tags_vector[:self.max_len]
                else:
                    padding_vector = torch.zeros(self.max_len-tags_vector.shape[0], 9)
                    tags_vector= torch.cat([tags_vector, padding_vector])
                    

                batch_tags.append(tags_vector)
            
            batch_tokens.append(tokens_vector)
            batch_ids.append(id)
            batch_lens.append(lens)
        
        # List[tensor] to tensor
        batch_tokens = torch.stack(batch_tokens)  # (b * max_len * embed_dim)
        if not is_test:
            batch_tags = torch.stack(batch_tags)  # (b * max_len * num_classes)

        return{
            'tokens': batch_tokens,
            'tags': batch_tags if not is_test else None,
            'id': batch_ids,
            'len': batch_lens,
        }
        

# if __name__ == '__main__':
#     try_token = "i'm a boy. i'd like to join you. Hellllllooooooooo~"
#     print(word_tokenize(try_token))


#     pass
#     with open('data\\slot\\train.json') as f:
#         data = json.load(f)

#     with open('cache\\slot\\vocab.pkl', 'rb') as f:
#         voc = pickle.load(f)  #　<class 'utils.Vocab'>

#     with open('cache\\slot\\tag2idx.json') as f:
#         label_mapping = json.load(f)

#     dataset = SeqTaggingClsDataset(
#         data=data,
#         label_mapping=label_mapping,
#         max_len= 15,
#         glove_path= 'glove.840B.300d.txt'
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=5,
#         shuffle=True,
#         collate_fn=dataset.collate_fn
#     )

#     for i, batch in enumerate(dataloader):
#         print(batch['tokens'].shape)
#         print(batch['tags'])
#         print(batch['id'])

#         if i == 10:
#             break
