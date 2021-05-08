import torch
from torch import nn, einsum
import torch.nn.functional as F
from tuframework.network_architecture.neural_network import SegmentationNetwork

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(SegmentationNetwork):
    def __init__(self,image_size,patch_size,num_classes,deep_supervision,dim,depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.do_ds = False
        norm_cfg = 'BN'
        activation_cfg = 'ReLU'
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0 and image_size[2] % patch_size[2] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size[0] // patch_size[0]) *(image_size[1] // patch_size[1]) *(image_size[2] // patch_size[2])
        patch_dim = channels * patch_size[0]* patch_size[1]* patch_size[2]
        #assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p0)(h p1) (w p2) -> b (d h w) (p0 p1 p2 c)', p0=patch_size[0], p1=patch_size[1],
                      p2=patch_size[2]),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,patch_dim*num_classes),
            Rearrange('b (d h w) (p0 p1 p2 c) -> b c (d p0)(h p1)(w p2)',d=8,h=8,w=8,p0 = patch_size[0], p1 = patch_size[1], p2 = patch_size[2]),
            #nn.Conv3d(patch_dim,num_classes,kernel_size=1)
        )

    def forward(self, img):
        seg_outputs=[]
        x = self.to_patch_embedding(img)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        seg_outputs.append(self.mlp_head(x))
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]
