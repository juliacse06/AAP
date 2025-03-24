import torch
from torch import nn
import torch.nn.functional as F
from self_attention import EncoderLayer

SMILESCLen = 64
PL_LEN = 32
PP_LEN = 58
hidden_dim = 384
out_channle = 384


class Squeeze(nn.Module):  # Dimention Module
    @staticmethod
    def forward(input_data: torch.Tensor):
        return input_data.squeeze()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 128
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        # SMILES, POCKET, PROTEIN Embedding
        self.ll_embed = nn.Embedding(SMILESCLen+1, embed_size)
        self.pl_embed = nn.Embedding(PL_LEN+1, embed_size)
        self.pp_embed = nn.Linear(PP_LEN, embed_size)

        
        self.conv_pp = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 8),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 12),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )
        

        self.conv_pl = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 8),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 12),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )

        
        self.conv_ll = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 6),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 8),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )
        
        ###################3 self-Attention Module
        self.sa_attention_poc = EncoderLayer(pkt_oc, pkt_oc, 0.1, 0.1, 2)
        self.adaptmaxpool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = Squeeze()

        # Dropout
        self.cat_dropout = nn.Dropout(0.2)
        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + pkt_oc + smi_oc, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, out_channle))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output

    def attention_net(self, x):
        q = torch.tanh(torch.matmul(x, self.w_omega))
        k = torch.tanh(torch.matmul(x, self.w_omega))
        v = torch.tanh(torch.matmul(x, self.w_omega))

        # Compute the attention output
        context = self.scaled_dot_product_attention(q, k, v)

        # Apply residual connection
        output = x + context

        return output
  

    def forward(self, data):
        pp, pl, ll = data
        
        # TODO:  PP Layers
        pp = pp.to(torch.float32)
        pp_embed = self.pp_embed(pp)
        pp_embed = torch.transpose(pp_embed, 1, 2)
        pp_conv = self.conv_pp(pp_embed)

        # TODO: PL layer
        pl = pl.to(torch.int32)
        pl_embed = self.pl_embed(pl)
        pl_embed = self.sa_attention_poc(pl_embed, pl_embed)
        pl_embed = torch.transpose(pl_embed, 1, 2)
        pl_conv = self.conv_pl(pl_embed)


        # TODO: LL Layer
        ll_embed = self.ll_embed(ll)
        ll_embed = torch.transpose(ll_embed, 1, 2)
        ll_conv = self.conv_ll(ll_embed)

        pp_conv = torch.reshape(pp_conv, (-1, 128))
        pl_conv = torch.reshape(pl_conv, (-1, 128))
        ll_conv = torch.reshape(ll_conv, (-1, 128))

        concat = torch.cat([pp_conv, pl_conv, ll_conv], dim=1)
        concat = torch.reshape(concat, (concat.shape[0], -1, 128*3))
        
        concat = self.attention_net(concat)
        concat = torch.reshape(concat, (-1, 128*3))
        #print(f"After Concat shape: {concat.shape}")
        concat = self.cat_dropout(concat)

        output = self.classifier(concat)
        return output