import torch
import math
mask = torch.zeros(1, 2, 3, 4, dtype=torch.bool).cuda()
print(mask)
assert mask is not None
not_mask = ~mask
print(not_mask)
d_embed = not_mask.cumsum(1, dtype=torch.float32)
y_embed = not_mask.cumsum(2, dtype=torch.float32)
x_embed = not_mask.cumsum(3, dtype=torch.float32)
eps = 1e-6
d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * 2*math.pi
y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * 2*math.pi
x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * 2*math.pi
print("d_embed",d_embed)
print("y_embed",y_embed)
print("x_embed",x_embed)
dim_tx = torch.arange(128, dtype=torch.float32,device=mask.device)
dim_tx = 10000 ** (3 * (dim_tx // 3) / 128)

dim_ty = torch.arange(128, dtype=torch.float32,device=mask.device)
dim_ty = 10000 ** (3 * (dim_ty // 3) / 128)

dim_td = torch.arange(128, dtype=torch.float32,device=mask.device)
dim_td = 10000 ** (3 * (dim_td // 3) / 128)
pos_x = x_embed[:, :, :, :, None] / dim_tx
pos_y = y_embed[:, :, :, :, None] / dim_ty
pos_d = d_embed[:, :, :, :, None] / dim_td
print("pos_x.shape",pos_x.shape)
print("pos_y.shape",pos_y.shape)
print("pos_d.shape",pos_d.shape)
pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
print("pos_x.shape",pos_x.shape)
print("pos_y.shape",pos_y.shape)
print("pos_d.shape",pos_d.shape)
pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
print("pos.shape",pos.shape)
