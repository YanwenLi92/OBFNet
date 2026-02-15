import torch
import torch.nn.functional as F
import models

def multi_loss(y_hat, y, all_filter, h_reir, win, context_len):

    lamda = 1
    sisnr = SI_SNR_LOSS(y_hat, y)
    reir = ReIR_Loss(y_hat, y, all_filter, h_reir, win, context_len)
    mulloss = lamda*sisnr+(1-lamda)*reir

    return mulloss, sisnr, reir


def SI_SNR_LOSS(y_hat, y):

    batch_size_est, nsample_est = y_hat.size()
    batch_size_ori, nsample_ori = y.size()

    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."

    # batch_size = batch_size_est
    # nsample = nsample_est

    # zero mean signals
    y_hat = y_hat - torch.mean(y_hat, 1, keepdim=True).expand_as(y_hat)
    y = y - torch.mean(y, 1, keepdim=True).expand_as(y)
    sisnrloss = 0 - calc_sdr(y_hat, y)

    return sisnrloss


def calc_sdr(estimation, origin):

    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)

    scale = torch.sum(origin * estimation, 1, keepdim=True) / origin_power  # (batch, 1)

    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)

    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)

    return torch.mean(10 * torch.log10(true_power) - 10 * torch.log10(res_power))  # mean(batch, 1) = 1


def ReIR_Loss(y_hat, y, all_filter, h_reir, win, context_len):

    # all_filter: B, nmic, num_spk=1, FrmNum, L; (L: length of IR)
    # y: B, T
    # h_reir: B,nmic*Lh; (Lh: length of ReIR)

    origin = y
    origin_power = torch.pow(origin, 2).sum(1, keepdim=True)  
    scale = origin_power / (torch.sum(origin * y_hat, 1, keepdim=True) + 1e-8)  
    all_filter = all_filter.squeeze(2)
    B, nmic, FrmNum, L = all_filter.size()  
    all_filter = scale.unsqueeze(-1).unsqueeze(-1) * all_filter
    Lh = h_reir.size(-1) // nmic
    h_reir = torch.cat([h_reir.view(B, nmic, Lh).unsqueeze(1)] * FrmNum, 1)
    Ed = torch.eye(Lh).to(y.device)
    uh_delay = Ed[:, context_len].unsqueeze(1)  # Lh,1
    h_reir = h_reir.transpose(1, 2).contiguous().view(B * nmic * FrmNum, -1)  
    pad = torch.zeros(B * nmic * FrmNum, L - 1).type(h_reir.type())
    h_reir_long = torch.cat([pad, h_reir], 1)
    h_multi_reir = F.conv1d(h_reir_long.view(1, -1, L - 1 + Lh), all_filter.view(-1, 1, L),
                            groups=B * nmic * FrmNum).squeeze().view(B, nmic, FrmNum, Lh)  
    uh_estimation = torch.sum(h_multi_reir, dim=1, keepdim=False)  
    UH = torch.fft.fft(uh_delay, dim=0)  # Lh,1
    H_multi_RTF = torch.fft.fft(uh_estimation, dim=2)  # B,FrmNum, Lh
    UH_all = torch.cat([UH.unsqueeze(0)] * B * FrmNum, 0).squeeze(-1).view(B, FrmNum, Lh)

    # simple TD-vad
    vad_thre = 5  # reverbClean
    input = y.unsqueeze(1)  # B, nmic=1, T
    _, all_mic_context, _ = models.seg_signal_context(input, win, context_len)  # B, 1, FrmNum, 2 * context + window
    Rn11 = torch.mean(torch.square(y[:, 0:120 * 8]), dim=1, keepdim=True).expand(B, FrmNum)  # B.
    Ry11 = torch.mean(torch.square(all_mic_context.squeeze(1)), dim=2, keepdim=False)  # B, FrmNum
    vad_ratio = Ry11 / Rn11
    vad_idx = vad_ratio <= vad_thre  # noise dominant

    loss_ = torch.square(
        torch.real(H_multi_RTF) - torch.real(UH_all)) + torch.square(torch.imag(H_multi_RTF) - torch.imag(UH_all))
    loss_ = torch.mean(loss_, dim=2, keepdim=False)
    vad_mask = (~vad_idx)
    loss_reir = loss_[vad_mask]

    return loss_reir.mean()


