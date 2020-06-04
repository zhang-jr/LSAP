import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCoherenceMetric(nn.Module):

    def __init__(self, feature_sequence):
        super(TemporalCoherenceMetric, self).__init__()
        self.feature_sequence = feature_sequence

    def similarity(self, x, dimension=2, metric='Euclidean'):
        if metric == 'Multiply':
            if dimension == 2:
                sim = torch.matmul(x, x.t())
            elif dimension == 3:
                sim = torch.matmul(x, x.transpose(2, 1))
        elif metric == 'Euclidean':
            M = x.size(0)
            xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(M, M)
            sim = xx + xx.t()
            sim.addmm_(1, -2, x, x.t())
            sim = sim.clamp(min=1e-12).sqrt()
        elif metric == 'cosine':
            sim = F.cosine_similarity(x, x)

        sim = torch.triu(sim, diagonal=1)
        mask = torch.triu(torch.ones_like(sim).byte(), diagonal=1)
        temporal_sim = -torch.masked_select(sim, mask)

        temporal_sim = F.softmax(temporal_sim, dim=0)
        temporal_info = torch.log(temporal_sim)
        temporal_entropy = (-temporal_sim * temporal_info).sum()
        return temporal_entropy.unsqueeze(dim=0)


    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-1])
        output = self.model(x)
        output = F.normalize(output)
        temporal_entropy = self.similarity(output)

        return temporal_entropy.unqueeze(dim=0)

