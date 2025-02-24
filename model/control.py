"""
This source code is modified from (https://github.com/SimiaoZuo/Transformer-Hawkes-Process)
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

PAD = 0

class MyLayerNorm2d(nn.Module):

    def __init__(self, channels):
        super(MyLayerNorm2d, self).__init__()
        self._scale = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self._offset = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        normalized = (x - x.mean(dim=[-3, -2, -1], keepdim=True)
                      ) / x.std(dim=[-3, -2, -1], keepdim=True)
        return self._scale * normalized + self._offset

def forward_fill_3d(t: torch.Tensor) -> torch.Tensor:
    n_batch, t_dim, n_dim = t.shape  # Handle 3D tensor: (batch, time, dim)
    rng = torch.arange(t_dim)
    rng_3d = rng.unsqueeze(0).unsqueeze(2).repeat(n_batch, 1, n_dim)
    rng_3d[t == 0] = 0
    idx = rng_3d.cummax(1).values
    filled_t = t[torch.arange(n_batch)[:, None, None], idx, torch.arange(n_dim)[None, None, :]]
    return filled_t


class Transformer_Encoder(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, args):
        super().__init__()

        width = args.state_dim
        device = args.device
        
        if args.dataset == 'pendulum':
            obs_embedder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Flatten(),
                nn.Linear(in_features=108, out_features=width),
                nn.ReLU()
                )
        elif args.dataset == 'physionet':
            obs_embedder = nn.Sequential(nn.Linear(2 * 37, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width))
        elif args.dataset == 'ushcn':
            obs_embedder = nn.Sequential(nn.Linear(2 * 5, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width))
            
        elif args.dataset == 'person_activity':
            obs_embedder = nn.Sequential(nn.Linear(2 * 12, width),
                                         nn.ReLU(),
                                         nn.LayerNorm(width),
                                         nn.Linear(width, width))
                    
        # OPTIONAL recurrent layer, this sometimes helps
        self.encoder = Encoder(args, width, device, obs_embedder)
        self.rnn = RNN_layers(width)


    def forward(self, obs, event_time, obs_mask=None, event_mask=None):
        if obs_mask is None:
            obs_mask = torch.ones_like(obs)
        if event_mask is None:
            event_mask = torch.ones_like(event_time)
        enc_output, y_observed = self.encoder(obs, event_time, obs_mask=obs_mask, event_mask=event_mask)
        enc_output = forward_fill_3d(enc_output)
        enc_output = self.rnn(enc_output, event_mask)

        return enc_output, y_observed



class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, width):
        super().__init__()

        self.gru = nn.GRU(width, width, num_layers=1, batch_first=True)
        self.linear = nn.Linear(width, width)

    def forward(self, inp, non_pad_mask):

        lengths = non_pad_mask.long().sum(1).cpu()
        lengths = torch.ones_like(lengths) * inp.size(1)

        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            inp, lengths, batch_first=True, enforce_sorted=False)
        temp = self.gru(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]
        out = self.linear(out)
        return out

class Decoder(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, args):
        super().__init__()

        self.dataset = args.dataset
        self.task = args.task
        self.ld = args.state_dim
        self.od= args.out_dim
        
        
        if self.dataset == 'pendulum':
            if self.task == 'regression':
                self.decoder = nn.Sequential(nn.Linear(self.ld, self.ld),
                                            nn.ReLU(),
                                            nn.Linear(self.ld, self.od))      
            elif self.task == 'interpolation':
                self.decoder = nn.Sequential(nn.Linear(in_features=self.ld, out_features=144),
                                             nn.ReLU(),
                                             nn.Unflatten(-1, [16, 3, 3]),
                                             
                                             nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=4, padding=2),
                                             MyLayerNorm2d(channels=16),
                                             nn.ReLU(),
                                             
                                             nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=3, stride=2, padding=1),
                                             MyLayerNorm2d(channels=12),
                                             nn.ReLU(),
                                             
                                             nn.ConvTranspose2d(in_channels=12, out_channels=1, kernel_size=2, stride=2, padding=5),
                                             nn.Sigmoid()
                                             )
                
        elif self.dataset == 'physionet':
            self.decoder = nn.Sequential(nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.od))
        elif self.dataset == 'ushcn':
            self.decoder = nn.Sequential(nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.ld),
                                         nn.ReLU(),
                                         nn.Linear(self.ld, self.od))
            
        elif self.dataset == 'person_activity':
            self.decoder = nn.Sequential(
                                         nn.Linear(self.ld, self.od),)           
    def forward(self, input):
        
        if self.dataset == 'pendulum' and self.task == 'interpolation':
            n, b, t, d = input.size()
            input = rearrange(input, 'n b t d -> (n b t) d')
            out = self.decoder(input)
            out = rearrange(out, '(n b t) x y z -> n b t x y z', n=n, b=b, t=t)
        else:
            out = self.decoder(input)
        return out
    
class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, args, width, device, obs_embedder):
        super().__init__()
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / width) for i in range(width)],
            device=device)
        
        self.obs_emb = obs_embedder
        self.dataset = args.dataset
        self.task = args.task
        self.info_type = args.info_type
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(width, width, 1, width, width, dropout=args.drop_out, normalize_before=False)
            for _ in range(args.n_layer)])


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])

        # return result# * non_pad_mask
        return result * non_pad_mask

    def forward(self, obs, event_time, obs_mask=None, event_mask=None):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding

        slf_attn_mask_subseq = get_subsequent_mask(obs_mask).bool()
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_mask, seq_q=event_mask)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        
        if self.task == "extrapolation":
            if self.info_type == "history":
                slf_attn_mask = ~(slf_attn_mask_keypad * ~slf_attn_mask_subseq).gt(0)
        else:
            if self.info_type == "history":
                slf_attn_mask = ~(slf_attn_mask_keypad * ~slf_attn_mask_subseq).gt(0)
            elif self.info_type == "full":
                slf_attn_mask = ~slf_attn_mask_keypad.gt(0)
            elif self.info_type == "future":
                slf_attn_mask_subseq = get_subsequent_mask(obs_mask, current=False).bool()
                b, t, _ = slf_attn_mask_subseq.size()
                diag = torch.ones(b, t).diag_embed().to(obs.device)
                diag = diag.type_as(slf_attn_mask_subseq)
                slf_attn_mask = ~(slf_attn_mask_keypad * slf_attn_mask_subseq).gt(0)

        non_pad_mask = event_mask.unsqueeze(-1)
        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        if self.dataset == "pendulum":
            input_obs = obs
            enc_ = []
            for i in range(obs.size(1)):
                h_in = input_obs[:, i].float()
                for layer in self.obs_emb:
                    h_in = layer(h_in)
                enc_o = h_in
                enc_.append(enc_o)
            enc_output = torch.stack(enc_, dim=1)
            
        else:
            input_obs = torch.cat([obs, obs_mask], dim=-1)
            b, t= input_obs.size(0), input_obs.size(1)
            input_obs = rearrange(input_obs, 'b t ... -> (b t) ...')
            enc_output = self.obs_emb(input_obs)
            enc_output = rearrange(enc_output, '(b t) d -> b t d', b=b, t=t)
        
        y_observed = enc_output.clone().detach()
        
        enc_output += tem_enc
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output, y_observed


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = seq_k.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk x dim
    return padding_mask


def get_subsequent_mask(seq, current=True):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    seq = seq.reshape(seq.size(0), seq.size(1), -1)

    sz_b, len_s, _ = seq.size()

    if current:
        # Include datas at current times for attention
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    else:  
        # Exclude datas at current times for attention
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
        
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask



class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, hidden_dim, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, hidden_dim, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x