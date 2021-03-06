import torch
import torch.nn as nn

class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim=100, emb_dim=300, emb_len=100,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1):
        super().__init__()
        self.embedding = nn.Embedding(emb_len, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = torch.sigmoid(self.predictor(feature))
        return preds
