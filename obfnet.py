import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import models, loss

# OBFNet

class OBFNet(nn.Module):
    def __init__(
        self,
        enc_dim      = 64,   # number of output channels of the encoder: N
        feature_dim  = 64,   # number of output channels of the bottleneck layer: B
        hidden_dim   = 128,  # number of hidden units in LSTM/BiLSTM
        layer        = 6,    # number of DPRNN block
        segment_size = 4,    # chunk size Kc for DPRNN
        nspk         = 1,    # single-speaker assumption
        win_len      = 16,   # window length in ms
        context_len  = 15,   # context length: C
        sr           = 8000, # sampling frequency in Hz
        micNum       = 4,    # number of microphones: M
        rnn_type     = 'LSTM',
        ref_ch       = 0,    # choose a reference channel
        Lh           = 512   # length of IR
        ):

        super(OBFNet, self).__init__()

        # hyper parameters
        self.num_spk = nspk  # 说话人个数
        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.win = int(sr * win_len / 1000)
        self.context = context_len
        self.stride = self.win // 2
        self.filter_dim = self.context * 2 + 1
        self.layer = layer
        self.segment_size = segment_size
        self.mic = micNum
        self.rnn_type = rnn_type
        self.eps = 1e-8
        self.ref_ch = ref_ch
        self.Lh = Lh
        self.online = True

        self.cnn1 = nn.Conv1d(1, self.mic * self.filter_dim, 1, stride=1, bias=False)  # trainable poles: a11,a12,...,a1L,a21,...aML
        self.hardTanh = nn.Hardtanh(0, 0.9999)

        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.win, bias=False)
        self.enc_LN = models.cLN(self.enc_dim, eps=1e-08)

        # weight estimator
        self.all_BF = models.BF_module(self.filter_dim + self.enc_dim, self.feature_dim, self.hidden_dim, self.filter_dim, self.num_spk, self.mic, self.layer, self.segment_size, self.online, self.rnn_type)


    def seq_cos_sim(self, ref, target):

        assert ref.size(1) == target.size(1), "Inputs should have same length."
        assert ref.size(2) >= target.size(2), "Reference input should be no smaller than the target input."

        seq_length = ref.size(1)

        larger_ch = ref.size(0)
        if target.size(0) > ref.size(0):
            ref = ref.expand(target.size(0), ref.size(1), ref.size(2)).contiguous()
            larger_ch = target.size(0)
        elif target.size(0) < ref.size(0):
            target = target.expand(ref.size(0), target.size(1), target.size(2)).contiguous()

        # L2 norms
        ref_norm = F.conv1d(ref.view(1, -1, ref.size(2)).pow(2),
                            torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
                            groups=larger_ch * seq_length)
        ref_norm = ref_norm.sqrt() + self.eps
        target_norm = target.norm(2, dim=2).view(1, -1, 1) + self.eps
        # cosine similarity
        cos_sim = F.conv1d(ref.view(1, -1, ref.size(2)),
                           target.view(-1, 1, target.size(2)),
                           groups=larger_ch * seq_length)
        cos_sim = cos_sim / (ref_norm * target_norm)

        return cos_sim.view(larger_ch, seq_length, -1)

    def FaS_OBF(self, input, all_filter, Apole):

        # input: batch_size, nmic, FrmNum, lenf = 2*contex + window
        # Apole: nmic, filter_dim
        # all_filter : batch_size, nmic, num_spk=1, FrmNum, filter_dim
        
        batch_size, _, FrmNum, lenf = input.size()

        A = torch.zeros(self.mic, self.filter_dim, self.filter_dim).type(input.type())
        B = torch.zeros(self.mic, self.filter_dim, 1).type(input.type())
        D = -Apole
        for ii in range(0, self.filter_dim):
            if ii == 0:
                B[:, ii, 0] = torch.sqrt(1 - torch.square(Apole[:, ii]))
            else:
                B[:, ii, 0] = torch.sqrt(1 - torch.square(Apole[:, ii])) * torch.prod(D[:, 0:ii], dim=1)
            for jj in range(0, ii + 1):
                if jj == ii:
                    A[:, ii, jj] = Apole[:, ii]
                elif jj == ii - 1:
                    A[:, ii, jj] = torch.sqrt(1 - torch.square(Apole[:, ii])) * torch.sqrt(
                        1 - torch.square(Apole[:, jj]))
                else:
                    A[:, ii, jj] = torch.sqrt(1 - torch.square(Apole[:, ii])) * torch.sqrt(
                        1 - torch.square(Apole[:, jj])) * torch.prod(D[:, jj + 1:ii], dim=1)
        AB = []
        for numnum in range(0, self.Lh):
            ab = torch.bmm(torch.linalg.matrix_power(A, self.Lh - 1 - numnum), B).unsqueeze(3)
            AB.append(ab)
        AB_w = torch.cat(AB, dim=3)  # nmic,L,1,Lh
        AB_w_ = torch.cat([AB_w.squeeze().unsqueeze(0)] * batch_size * FrmNum * self.num_spk, 0)  # batch*FrmNum*num_spk, nmic,L,Lh
        all_filter_ = all_filter.permute(0, 3, 2, 1, 4).contiguous().view(batch_size * FrmNum * self.num_spk * self.mic, 1, self.filter_dim)
        AB_h_all = torch.bmm(all_filter_, AB_w_.view(batch_size * FrmNum * self.num_spk * self.mic, self.filter_dim, self.Lh))  # batch*FrmNum*num_spk*nmic,1,Lh
        all_filter_OBF = AB_h_all.squeeze().view(batch_size, FrmNum, self.num_spk, self.mic, self.Lh).permute(0, 2, 3, 1, 4).contiguous()  # B,nspk,nmic,FrmNum,Lh

        # convolve with all mic context segments
        pad = torch.zeros(batch_size, self.mic, FrmNum, self.Lh - self.filter_dim).type(input.type())
        input_ = torch.cat([pad, input], 3)  # Batch,nmic,FrmNum, Lh - 1 + window = lenf + Lh -L
        y_context = torch.cat([input_.unsqueeze(1)] * self.num_spk, 1)  # B, nspk, nmic, FrmNum, Lh - 1 + window
        y_context_saved = y_context.view(batch_size * self.num_spk, self.mic, FrmNum, self.Lh - 1 + self.win)  # B*nspk, nmic, FrmNum, Lh - 1 + window
        all_output = F.conv1d(y_context_saved.view(1, -1, self.Lh - 1 + self.win), all_filter_OBF.view(-1, 1, self.Lh),
                                groups=batch_size * self.num_spk * self.mic * FrmNum)  # 1, B*nspk*nmic*FrmNum, win
        all_bf_output = all_output.view(batch_size * self.num_spk, self.mic, FrmNum, self.win)  # B*nspk, nmic, FrmNum, win

        return all_bf_output, all_filter_OBF

    def forward(self, input):  # B,T,nmic

        input = input.transpose(1, 2).contiguous()  # B,nmic,T
        batch_size = input.size(0)

        # split input into frames
        # all_seg → center_frame：B, nmic, FrmNum, window
        # all_mic_context → chunks：B, nmic, FrmNum, 2 * context + window
        # rest
        all_seg, all_mic_context, rest = models.seg_signal_context(input, self.win, self.context)
        seq_length = all_seg.size(2)  # FrmNum

        inputUnit = torch.ones(1, 1, 1).type(input.type())
        Apole = self.hardTanh(self.cnn1(inputUnit).squeeze().view(self.mic, self.filter_dim))

        # calculate cosine similarity
        all_context = all_mic_context  # B, nmic, FrmNum, 2 * context + window
        all_context_saved = all_context.view(-1, 1, self.context * 2 + self.win)  # B*nmic*FrmNum, 1, 2*context+window
        all_context = all_context.transpose(0, 1).contiguous().view(self.mic, -1, self.context * 2 + self.win)  # nmic, B*FrmNum, 2*context+window
        ref_segment = all_seg[:, self.ref_ch].contiguous().view(1, -1, self.win)  # 1, B*FrmNum, window
        all_cos_sim = self.seq_cos_sim(all_context, ref_segment)  # nmic, B*FrmNum, 2*context+1
        all_cos_sim = all_cos_sim.view(self.mic, batch_size, seq_length, self.filter_dim)  # nmic, B, FrmNum, 2*context+1
        all_cos_sim = all_cos_sim.permute(1, 0, 3, 2).contiguous().view(-1, self.filter_dim, seq_length)  # B*nmic, 2*context+1, FrmNum

        all_feature = self.encoder(all_context_saved).view(-1, seq_length, self.enc_dim)  # B*nmic, FrmNum, enc_dim
        all_feature = all_feature.transpose(1, 2).contiguous()  # B*nmic, enc_dim, FrmNum
        all_filter = self.all_BF(torch.cat([self.enc_LN(all_feature), all_cos_sim], 1))  # B, nmic, num_spk=1, FrmNum, filter_dim

        all_bf_output, all_filter_OBF = self.FaS_OBF(all_mic_context, all_filter, Apole)

        # reshape to utterance
        bf_signal = all_bf_output.view(batch_size * self.num_spk * self.mic, -1, self.win * 2)
        bf_signal1 = bf_signal[:, :, :self.win].contiguous().view(batch_size * self.num_spk * self.mic, 1, -1)[:, :, self.stride:]
        bf_signal2 = bf_signal[:, :, self.win:].contiguous().view(batch_size * self.num_spk * self.mic, 1, -1)[:, :, :-self.stride]
        bf_signal = bf_signal1 + bf_signal2  # B*nspk*nmic, 1, T
        if rest > 0:
            bf_signal = bf_signal[:, :, :-rest]

        bf_signal = bf_signal.view(batch_size, self.num_spk, self.mic, -1)  # B, nspk, nmic, T
        bf_signal = bf_signal.mean(2)  # B, nspk, T

        if self.num_spk == 1:
            bf_signal = bf_signal.squeeze(1)

        return bf_signal, all_filter_OBF.transpose(1, 2).contiguous()


def test_model(model):
    x = torch.rand(2, 24000, 4)  # (batch, length, num_mic)
    y, h = model(x)
    print(y.size())


if __name__ == "__main__":
    model = OBFNet(
        enc_dim      = 64,
        feature_dim  = 64,
        hidden_dim   = 128,
        layer        = 6,
        segment_size = 4,
        nspk         = 1,
        win_len      = 10,
        context_len  = 15,
        sr           = 8000,
        micNum       = 4,
        rnn_type     = 'LSTM',
        ref_ch       = 0,
        Lh           = 110
    )

    test_model(model)
