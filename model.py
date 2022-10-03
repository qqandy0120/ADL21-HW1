from typing import Dict

import torch
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 30,
        num_layers: int = 2,
        num_classes: int = 150,
        batch_first: bool = True,
        biderectional: bool = True,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # TODO: model architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.biderectional = biderectional
        
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bidirectional=self.biderectional,
        )
        '''
        input: 24(B) x  15(text_length) x 300(embedding_dim)

        2 outputs:
            output:  24(B) x 15(text_length) x 30(hidden_size)
            h0    :  2(num_layer) x 24(B) x 30(hidden_size)
        '''
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_classes
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, input) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # initializing hidden state for the first input with 0
        h0 = torch.zeros(self.num_layers, input.shape[0], self.hidden_size)
        output, _ = self.gru(input, h0.detach())

        output = output[:, -1, :]

        output = self.fc(output)

        return output



class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError


if __name__ == "__main__":
    pass
    # print(torch.__version__)
    # print(torch.cuda.is_available())


    # model = SeqClassifier(biderectional=False)
    # input = torch.randn(24,15,300)


    # outputs = model(input)

    # print(outputs.shape)




