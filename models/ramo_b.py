import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_nodes_nms(heatmap, threshold=0.15, radius=8, margin=20):
    """
    Extrai coordenadas de picos no heatmap usando Non-Maximum Suppression.
    """
    kernel = radius * 2 + 1
    local_max = F.max_pool2d(heatmap, kernel, stride=1, padding=kernel//2)
    peaks = (heatmap == local_max) & (heatmap > threshold)
    
    # Remover bordas para evitar artefatos
    if margin > 0:
        peaks[:, :, :margin, :] = False
        peaks[:, :, -margin:, :] = False
        peaks[:, :, :, :margin] = False
        peaks[:, :, :, -margin:] = False
        
    y, x = torch.where(peaks[0, 0])
    return torch.stack([x, y], dim=1).float()

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GeometryDecoder(nn.Module):
    """
    Arquitetura compativel com o checkpoint cityscale/spacenet do SAM-Road.
    """
    def __init__(self, in_channels=256):
        super().__init__()
        activation = nn.GELU
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Saida bruta (logits) para ser processada pelo Sigmoid na inferencia
        return self.decoder(x)
