import torch
import torch.nn as nn
import config


class CombinedLoss(nn.Module):
    def __init__(
        self,
        onset_pos_weight=None,
        fret_ignore_index=None,
        onset_loss_weight=1.0,
        fret_loss_weight=1.0,
    ):
        super().__init__()
        self.onset_loss_weight = onset_loss_weight
        self.fret_loss_weight = fret_loss_weight
        self.criterion_onset = nn.BCEWithLogitsLoss(pos_weight=onset_pos_weight, reduction='none')
        self.criterion_fret = nn.CrossEntropyLoss(
            ignore_index=fret_ignore_index if fret_ignore_index is not None else config.FRET_PADDING_VALUE
        )

    def forward(self, onset_logits, fret_logits, onset_targets, fret_targets, lengths):
        loss_o_unreduced = self.criterion_onset(onset_logits, onset_targets)

        batch_size, max_len, num_strings = onset_logits.shape
        mask_onset = torch.zeros_like(onset_logits, dtype=torch.bool, device=onset_logits.device)
        for i in range(batch_size):
            mask_onset[i, :lengths[i], :] = True

        loss_o_masked = loss_o_unreduced * mask_onset.float()
        loss_o = loss_o_masked.sum() / mask_onset.sum().clamp(min=1)

        fret_logits_permuted = fret_logits.permute(0, 3, 1, 2)
        loss_f = self.criterion_fret(fret_logits_permuted, fret_targets)

        total_loss = (self.onset_loss_weight * loss_o) + (self.fret_loss_weight * loss_f)
        return total_loss, loss_o, loss_f