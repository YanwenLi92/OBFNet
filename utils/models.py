import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
reference:

2019 Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation
    GitHub: https://github.com/naplab/Conv-TasNet
    
2020 End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation
    GitHub: https://github.com/yluo42/TAC
    
"""


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


# dual-path RNN with transform-average-concatenate (TAC)
class DPRNN_TAC(nn.Module):
    """
    Deep duaL-path RNN with transform-average-concatenate (TAC) applied to each layer/block.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN_TAC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # DPRNN + TAC for 3D input (ch, N, T)
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])

        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.ch_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.ch_transform.append(nn.Sequential(nn.Linear(input_size, hidden_size * 3),
                                                   nn.PReLU()
                                                   )
                                     )
            self.ch_average.append(nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 3),
                                                 nn.PReLU()
                                                 )
                                   )
            self.ch_concat.append(nn.Sequential(nn.Linear(hidden_size * 6, input_size),
                                                nn.PReLU()
                                                )
                                  )
            if bidirectional:
                # default is to use noncausal LayerNorm for inter-chunk RNN and TAC modules. For causal setting change them to causal normalization techniques accordingly.
                self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.ch_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                self.row_norm.append(cLN(self.input_size, eps=1e-08))
                self.col_norm.append(cLN(self.input_size, eps=1e-08))
                self.ch_norm.append(cLN(self.input_size, eps=1e-08))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                    )

    def forward(self, input):
        # input shape: batch, ch, N, dim1, dim2
        # num_mic shape: batch,
        # apply RNN on dim1 first, then dim2, then ch

        batch_size, ch, N, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            # intra-segment RNN
            output = output.view(batch_size * ch, N, dim1, dim2)  # B*ch, N, dim1, dim2
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * ch * dim2, dim1,
                                                                     -1)  # B*ch*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*ch*dim2, dim1, N
            row_output = row_output.view(batch_size * ch, dim2, dim1, -1).permute(0, 3, 2,
                                                                                  1).contiguous()  # B*ch, N, dim1, dim2
            if self.bidirectional:
                row_output = self.row_norm[i](row_output)
            else:
                row_output = self.row_norm[i](
                    row_output.transpose(2, 3).contiguous().view(batch_size * ch, -1, dim2 * dim1)).view(batch_size * ch, -1, dim2, dim1).transpose(2, 3).contiguous()
            output = output + row_output  # B*ch, N, dim1, dim2

            # inter-segment RNN
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * ch * dim1, dim2, -1)  # B*ch*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, N
            col_output = col_output.view(batch_size * ch, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B*ch, N, dim1, dim2
            if self.bidirectional:
                col_output = self.col_norm[i](col_output)
            else:
                col_output = self.col_norm[i](
                    col_output.transpose(2, 3).contiguous().view(batch_size * ch, -1, dim2 * dim1)).view(batch_size * ch, -1, dim2, dim1).transpose(2, 3).contiguous()
            output = output + col_output  # B*ch, N, dim1, dim2

            # TAC for cross-channel communication
            ch_input = output.view(input.shape)  # B, ch, N, dim1, dim2
            ch_input = ch_input.permute(0, 3, 4, 1, 2).contiguous().view(-1, N)  # B*dim1*dim2*ch, N
            ch_output = self.ch_transform[i](ch_input).view(batch_size, dim1 * dim2, ch, -1)  # B, dim1*dim2, ch, H
            # fixed geometry array
            ch_mean = ch_output.mean(2).view(batch_size * dim1 * dim2, -1)  # B*dim1*dim2, H

            ch_output = ch_output.view(batch_size * dim1 * dim2, ch, -1)  # B*dim1*dim2, ch, H
            ch_mean = self.ch_average[i](ch_mean).unsqueeze(1).expand_as(ch_output).contiguous()  # B*dim1*dim2, ch, H
            ch_output = torch.cat([ch_output, ch_mean], 2)  # B*dim1*dim2, ch, 2H
            ch_output = self.ch_concat[i](ch_output.view(-1, ch_output.shape[-1]))  # B*dim1*dim2*ch, N
            ch_output = ch_output.view(batch_size, dim1, dim2, ch, -1).permute(0, 3, 4, 1, 2).contiguous()  # B, ch, N, dim1, dim2
            ch_output = ch_output.view(batch_size * ch, N, dim1, dim2)  # B*ch, N, dim1, dim2
            if self.bidirectional:
                ch_output = self.ch_norm[i](ch_output)  # B*ch, N, dim1, dim2
            else:
                ch_output = self.ch_norm[i](ch_output.transpose(2, 3).contiguous().view(batch_size * ch, -1, dim2 * dim1)).view(batch_size * ch, -1, dim2, dim1).transpose(2, 3).contiguous()
            output = output + ch_output

        output = self.output(output.transpose(2, 3).contiguous()).transpose(2, 3).contiguous()  # B*ch, N*nspk, dim1, dim2

        return output


class BF_module(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk, micNum, layer, segment_size, online, rnn_type):
        super(BF_module, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.mic = micNum
        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        if not online:
            bidirectional = True
        else:
            bidirectional = False
        self.DPRNN = DPRNN_TAC(rnn_type, self.feature_dim, self.hidden_dim, self.feature_dim*self.num_spk, num_layers=layer, bidirectional=bidirectional)

        # output layer
        self.output = nn.Conv1d(self.feature_dim, output_dim, 1)
        self.pRLU = nn.PReLU()

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):

        batch_size_ch, N, seq_length = input.shape
        batch_size = batch_size_ch//self.mic
        ch = self.mic
        enc_feature = self.BN(input)

        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)

        # pass to DPRNN
        enc_segments = enc_segments.view(batch_size, ch, -1, enc_segments.shape[2], enc_segments.shape[3])
        output = self.DPRNN(enc_segments).view(batch_size * ch * self.num_spk, self.feature_dim, self.segment_size, -1)

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)

        bf_filter = self.pRLU(8e-3 * self.output(output))  # B*micNum*num_spk, output_dim, nchunk
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, ch, self.num_spk, -1, self.output_dim)  # B, micNum, num_spk, nchunk, output_dim

        return bf_filter


def pad_input(input, window):
    """
    Zero-padding input according to window/stride size.
    """
    batch_size, micNum, nsample = input.shape
    stride = window // 2

    # pad the signals at the end for matching the window/stride size
    rest = window - (stride + nsample % window) % window
    if rest > 0:
        pad = torch.zeros(batch_size, micNum, rest).type(input.type())
        input = torch.cat([input, pad], 2)
    pad_aux = torch.zeros(batch_size, micNum, stride).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest

def seg_signal_context(x, window, context):
    """
    Segmenting the signal into chunks with specific context.
    input:
        x: size (B, ch, T)
        window: int
        context: int

    """

    # pad input accordingly
    # first pad according to window size
    input, rest = pad_input(x, window)
    batch_size, micNum, nsample = input.shape
    stride = window // 2

    # pad another context size
    pad_context = torch.zeros(batch_size, micNum, context).type(input.type())
    input = torch.cat([pad_context, input, pad_context], 2)  # B, ch, L

    # calculate index for each chunk
    nchunk = 2 * nsample // window - 1
    begin_idx = np.arange(nchunk) * stride
    begin_idx = torch.from_numpy(begin_idx).type(input.type()).long().view(1, 1, -1)  # 1, 1, nchunk
    begin_idx = begin_idx.expand(batch_size, micNum, nchunk)  # B, ch, nchunk
    # select entries from index
    chunks = [torch.gather(input, 2, begin_idx + i).unsqueeze(3) for i in
                range(2 * context + window)]  # B, ch, nchunk, 1
    chunks = torch.cat(chunks, 3)  # B, ch, nchunk, chunk_size

    # center frame
    center_frame = chunks[:, :, :, context:context + window]

    return center_frame, chunks, rest