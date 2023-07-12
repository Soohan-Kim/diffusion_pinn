import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F
#from torch.autograd import Function
#from torch.utils.cpp_extension import load
import os
import string

'''
Layers to support main network

classes, functions from following:
https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
https://github.com/yang-song/score_sde_pytorch/blob/main/models/up_or_down_sampling.py
https://github.com/yang-song/score_sde_pytorch/blob/main/op/upfirdn2d.py
https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py

'''

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  #conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  #nn.init.zeros_(conv.bias)
  return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  #conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  #nn.init.zeros_(conv.bias)
  return conv

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(torch.zeros((in_dim, num_units)), requires_grad=True) #nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)

class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block."""
    
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels//4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)
        
# def _setup_kernel(k):
#   k = np.asarray(k, dtype=np.float32)
#   if k.ndim == 1:
#     k = np.outer(k, k)
#   k /= np.sum(k)
#   assert k.ndim == 2
#   assert k.shape[0] == k.shape[1]
#   return k

# module_path = os.path.dirname(__file__)
# upfirdn2d_op = load(
#     "upfirdn2d",
#     sources=[
#         os.path.join(module_path, "upfirdn2d.cpp"),
#         os.path.join(module_path, "upfirdn2d_kernel.cu"),
#     ],
# )

# class UpFirDn2dBackward(Function):
#     @staticmethod
#     def forward(
#         ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
#     ):

#         up_x, up_y = up
#         down_x, down_y = down
#         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

#         grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

#         grad_input = upfirdn2d_op.upfirdn2d(
#             grad_output,
#             grad_kernel,
#             down_x,
#             down_y,
#             up_x,
#             up_y,
#             g_pad_x0,
#             g_pad_x1,
#             g_pad_y0,
#             g_pad_y1,
#         )
#         grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

#         ctx.save_for_backward(kernel)

#         pad_x0, pad_x1, pad_y0, pad_y1 = pad

#         ctx.up_x = up_x
#         ctx.up_y = up_y
#         ctx.down_x = down_x
#         ctx.down_y = down_y
#         ctx.pad_x0 = pad_x0
#         ctx.pad_x1 = pad_x1
#         ctx.pad_y0 = pad_y0
#         ctx.pad_y1 = pad_y1
#         ctx.in_size = in_size
#         ctx.out_size = out_size

#         return grad_input

#     @staticmethod
#     def backward(ctx, gradgrad_input):
#         kernel, = ctx.saved_tensors

#         gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

#         gradgrad_out = upfirdn2d_op.upfirdn2d(
#             gradgrad_input,
#             kernel,
#             ctx.up_x,
#             ctx.up_y,
#             ctx.down_x,
#             ctx.down_y,
#             ctx.pad_x0,
#             ctx.pad_x1,
#             ctx.pad_y0,
#             ctx.pad_y1,
#         )
#         # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
#         gradgrad_out = gradgrad_out.view(
#             ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
#         )

#         return gradgrad_out, None, None, None, None, None, None, None, None

# class UpFirDn2d(Function):
#     @staticmethod
#     def forward(ctx, input, kernel, up, down, pad):
#         up_x, up_y = up
#         down_x, down_y = down
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad

#         kernel_h, kernel_w = kernel.shape
#         batch, channel, in_h, in_w = input.shape
#         ctx.in_size = input.shape

#         input = input.reshape(-1, in_h, in_w, 1)

#         ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

#         out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
#         out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
#         ctx.out_size = (out_h, out_w)

#         ctx.up = (up_x, up_y)
#         ctx.down = (down_x, down_y)
#         ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

#         g_pad_x0 = kernel_w - pad_x0 - 1
#         g_pad_y0 = kernel_h - pad_y0 - 1
#         g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
#         g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

#         ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

#         out = upfirdn2d_op.upfirdn2d(
#             input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
#         )
#         # out = out.view(major, out_h, out_w, minor)
#         out = out.view(-1, channel, out_h, out_w)

#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         kernel, grad_kernel = ctx.saved_tensors

#         grad_input = UpFirDn2dBackward.apply(
#             grad_output,
#             kernel,
#             grad_kernel,
#             ctx.up,
#             ctx.down,
#             ctx.pad,
#             ctx.g_pad,
#             ctx.in_size,
#             ctx.out_size,
#         )

#         return grad_input, None, None, None, None

# def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):

#     out = UpFirDn2d.apply(
#         input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
#     )

#     return out
        
# def upsample_2d(x, k=None, factor=2, gain=1):
#     assert isinstance(factor, int) and factor >= 1
#     if k is None:
#         k = [1] * factor
#     k = _setup_kernel(k) * (gain * (factor ** 2))
#     p = k.shape[0] - factor
#     return upfirdn2d(x, torch.tensor(k, device=x.device),
#                     up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))
    
# def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
#     assert isinstance(factor, int) and factor >= 1

#     # Check weight shape.
#     assert len(w.shape) == 4
#     convH = w.shape[2]
#     convW = w.shape[3]
#     inC = w.shape[1]
#     outC = w.shape[0]

#     assert convW == convH

#     # Setup filter kernel.
#     if k is None:
#         k = [1] * factor
#     k = _setup_kernel(k) * (gain * (factor ** 2))
#     p = (k.shape[0] - factor) - (convW - 1)

#     stride = (factor, factor)

#     # Determine data dimensions.
#     stride = [1, 1, factor, factor]
#     output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
#     output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
#                         output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
#     assert output_padding[0] >= 0 and output_padding[1] >= 0
#     num_groups = _shape(x, 1) // inC

#     # Transpose weights.
#     w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
#     w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
#     w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

#     x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
#     ## Original TF code.
#     # x = tf.nn.conv2d_transpose(
#     #     x,
#     #     w,
#     #     output_shape=output_shape,
#     #     strides=stride,
#     #     padding='VALID',
#     #     data_format=data_format)
#     ## JAX equivalent

#     return upfirdn2d(x, torch.tensor(k, device=x.device),
#                     pad=((p + 1) // 2 + factor - 1, p // 2 + 1))
    
# def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
#     assert isinstance(factor, int) and factor >= 1
#     _outC, _inC, convH, convW = w.shape
#     assert convW == convH
#     if k is None:
#         k = [1] * factor
#     k = _setup_kernel(k) * gain
#     p = (k.shape[0] - factor) + (convW - 1)
#     s = [factor, factor]
#     x = upfirdn2d(x, torch.tensor(k, device=x.device),
#                     pad=((p + 1) // 2, p // 2))
#     return F.conv2d(x, w, stride=s, padding=0)
    
# class Conv2d(nn.Module):
#   """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

#   def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
#                resample_kernel=(1, 3, 3, 1),
#                use_bias=True,
#                kernel_init=None):
#     super().__init__()
#     assert not (up and down)
#     assert kernel >= 1 and kernel % 2 == 1
#     self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
#     if kernel_init is not None:
#       self.weight.data = kernel_init(self.weight.data.shape)
#     if use_bias:
#       self.bias = nn.Parameter(torch.zeros(out_ch))

#     self.up = up
#     self.down = down
#     self.resample_kernel = resample_kernel
#     self.kernel = kernel
#     self.use_bias = use_bias

#   def forward(self, x):
#     if self.up:
#       x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
#     elif self.down:
#       x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
#     else:
#       x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

#     if self.use_bias:
#       x = x + self.bias.reshape(1, -1, 1, 1)

#     return x
        
class Upsamplepp(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir and with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch)
        # elif with_conv:
        #     self.Conv2d_0 = Conv2d(in_ch, out_ch,
        #                                             kernel=3, up=True,
        #                                             resample_kernel=fir_kernel,
        #                                             use_bias=True,
        #                                             kernel_init=default_init())
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        # else:
        #     if not self.with_conv:
        #         h = upsample_2d(x, self.fir_kernel, factor=2)
        #     else:
        #         h = self.Conv2d_0(x)

        return h
    
# def downsample_2d(x, k=None, factor=2, gain=1):
#     assert isinstance(factor, int) and factor >= 1
#     if k is None:
#         k = [1] * factor
#     k = _setup_kernel(k) * gain
#     p = k.shape[0] - factor
#     return upfirdn2d(x, torch.tensor(k, device=x.device),
#                     down=factor, pad=((p + 1) // 2, p // 2))
    
class Downsamplepp(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        # else:
        #     if with_conv:
        #         self.Conv2d_0 = Conv2d(in_ch, out_ch,
        #                                             kernel=3, down=True,
        #                                             resample_kernel=fir_kernel,
        #                                             use_bias=True,
        #                                             kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        # else:
        #     if not self.with_conv:
        #         x = downsample_2d(x, self.fir_kernel, factor=2)
        #     else:
        #         x = self.Conv2d_0(x)

        return x
    
def naive_upsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))
    
class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      #self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      #nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
      if self.fir:
        pass
        # h = upsample_2d(h, self.fir_kernel, factor=2)
        # x = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = naive_upsample_2d(h, factor=2)
        x = naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        pass
        # h = downsample_2d(h, self.fir_kernel, factor=2)
        # x = downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = naive_downsample_2d(h, factor=2)
        x = naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      #print('temb', temb.size(), 'h', h.size(), self.Dense_0.weight.shape)
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)