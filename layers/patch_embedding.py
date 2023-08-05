# patch embedding
# Add CLS Token
# position embedding

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # patch embedding
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # nn.Parameter = 학습 가능한 파라미터로 설정하는 것임.
        # Add CLS Token
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        
        # position embedding
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)  # cls token을 x의 첫번째 차원으로 반복함.

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)  # torch.cat = concate  -> cls_tokens 와 x 를 연결함. (= cls 토큰 추가 과정.)
        
        # add position embedding
        x += self.positions
        return x