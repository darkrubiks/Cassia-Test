import torch
import torch.nn.functional as F

import math


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits
    
class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

class CosineClassifier(torch.nn.Module):
    """
    Turns an embedding into a cosine-similarity logit matrix (B, num_classes).
    No bias; weights are ℓ2-normalised every forward call.
    """
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, embedding_dim)  -->  (B, num_classes)
        x = F.normalize(x)                      # (B, d)
        w = F.normalize(self.weight)            # (C, d)
        return F.linear(x, w)                   # cosine similarities


class FaceModel(torch.nn.Module):
    def __init__(self, backbone, classifier, arcface):
        super().__init__()
        self.backbone   = backbone
        self.classifier = classifier
        self.arcface    = arcface

    def forward(self, x, labels=None):
        emb  = self.backbone(x)
        cos  = self.classifier(emb)
        if labels is None:               # inference path
            return cos
        return self.arcface(cos, labels) # training path
