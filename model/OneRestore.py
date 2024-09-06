# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchvision import transforms
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numbers

from thop import profile
import numpy as np
import time
from torchvision import transforms


class OneRestore(nn.Module):
	def __init__(self, channel = 32):
		super(OneRestore,self).__init__()
		self.norm = lambda x: (x-0.5)/0.5
		self.denorm = lambda x: (x+1)/2
		self.in_conv = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.encoder = encoder(channel)
		self.middle = backbone(channel)
		self.decoder = decoder(channel)
		self.out_conv = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)

	def forward(self,x,embedding):
		x_in = self.in_conv(self.norm(x))
		x_l, x_m, x_s, x_ss = self.encoder(x_in, embedding)
		x_mid = self.middle(x_ss, embedding)
		x_out = self.decoder(x_mid, x_ss, x_s, x_m, x_l, embedding)
		out = self.out_conv(x_out) + x
		return self.denorm(out)

class encoder(nn.Module):
	def __init__(self,channel):
		super(encoder,self).__init__()    

		self.el = ResidualBlock(channel)#16
		self.em = ResidualBlock(channel*2)#32
		self.es = ResidualBlock(channel*4)#64
		self.ess = ResidualBlock(channel*8)#128

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#16 32
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#32 64
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 128
		self.conv_esstesss = nn.Conv2d(8*channel,16*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 256

	def forward(self,x,embedding):

		elout = self.el(x, embedding)#16
		x_emin = self.conv_eltem(self.maxpool(elout))#32
		emout = self.em(x_emin, embedding)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin, embedding)
		x_esin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_esin, embedding)#128

		return elout, emout, esout, essout#,esssout

class backbone(nn.Module):
	def __init__(self,channel):
		super(backbone,self).__init__()    

		self.s1 = ResidualBlock(channel*8)#128
		self.s2 = ResidualBlock(channel*8)#128

	def forward(self,x,embedding):

		share1 = self.s1(x, embedding)
		share2 = self.s2(share1, embedding)

		return share2

class decoder(nn.Module):
	def __init__(self,channel):
		super(decoder,self).__init__()    

		self.dss = ResidualBlock(channel*8)#128
		self.ds = ResidualBlock(channel*4)#64
		self.dm = ResidualBlock(channel*2)#32
		self.dl = ResidualBlock(channel)#16

		#self.conv_dssstdss = nn.Conv2d(16*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#256 128
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 64
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 32
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)#32 16

	def _upsample(self,x,y):
		_,_,H0,W0 = y.size()
		return F.interpolate(x,size=(H0,W0),mode='bilinear')

	def forward(self, x, x_ss, x_s, x_m, x_l, embedding):

		dssout = self.dss(x + x_ss, embedding)
		x_dsin = self.conv_dsstds(self._upsample(dssout, x_s))        
		dsout = self.ds(x_dsin + x_s, embedding)
		x_dmin = self.conv_dstdm(self._upsample(dsout, x_m))
		dmout = self.dm(x_dmin + x_m, embedding)
		x_dlin = self.conv_dmtdl(self._upsample(dmout, x_l))
		dlout = self.dl(x_dlin + x_l, embedding)

		return dlout


class ResidualBlock(nn.Module):  # Edge-oriented Residual Convolution Block 面向边缘的残差网络块 解决梯度消失的问题
	def __init__(self, channel, norm=False):
		super(ResidualBlock, self).__init__()

		self.el = TransformerBlock(channel, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

	def forward(self, x,embedding):
		return self.el(x,embedding)

def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)
		assert len(normalized_shape) == 1
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)
		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
	def __init__(self, dim, LayerNorm_type):
		super(LayerNorm, self).__init__()
		if LayerNorm_type == 'BiasFree':
			self.body = BiasFree_LayerNorm(dim)
		else:
			self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		h, w = x.shape[-2:]
		return to_4d(self.body(to_3d(x)), h, w)

class Cross_Attention(nn.Module):
    def __init__(self, 
		 		dim, 
				num_heads, 
				bias,
				q_dim = 324):
        super(Cross_Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        sqrt_q_dim = int(math.sqrt(q_dim))
        self.resize = transforms.Resize([sqrt_q_dim, sqrt_q_dim])
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(q_dim, q_dim, bias=bias)
        
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x, query):
        b,c,h,w = x.shape

        q = self.q(query)
        k, v = self.kv_dwconv(self.kv(x)).chunk(2, dim=1) 
        k = self.resize(k)
	
        q = repeat(q, 'b l -> b head c l', head=self.num_heads, c=self.dim//self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Self_Attention(nn.Module):
    def __init__(self, 
		 		dim, 
				num_heads, 
				bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
	def __init__(self, 
	      		dim, 
				ffn_expansion_factor, 
				bias):
		super(FeedForward, self).__init__()

		hidden_features = int(dim * ffn_expansion_factor)

		self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

		self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

		self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x

class TransformerBlock(nn.Module):
	def __init__(self, 
	      		dim, 
				num_heads=8, 
				ffn_expansion_factor=2.66, 
				bias=False, 
				LayerNorm_type='WithBias'):
		super(TransformerBlock, self).__init__()
		self.norm1 = LayerNorm(dim, LayerNorm_type)
		self.cross_attn = Cross_Attention(dim, num_heads, bias)
		self.norm2 = LayerNorm(dim, LayerNorm_type)
		self.self_attn = Self_Attention(dim, num_heads, bias)
		self.norm3 = LayerNorm(dim, LayerNorm_type)
		self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

	def forward(self, x, query):
		x = x + self.cross_attn(self.norm1(x),query)
		x = x + self.self_attn(self.norm2(x))
		x = x + self.ffn(self.norm3(x))
		return x

if __name__ == '__main__':
	net = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
	# x = torch.Tensor(np.random.random((2,3,256,256))).to("cuda" if torch.cuda.is_available() else "cpu")
	# query = torch.Tensor(np.random.random((2, 324))).to("cuda" if torch.cuda.is_available() else "cpu")
	# out = net(x, query)
	# print(out.shape)
	input = torch.randn(1, 3, 512, 512).to("cuda" if torch.cuda.is_available() else "cpu")
	query = torch.Tensor(np.random.random((1, 324))).to("cuda" if torch.cuda.is_available() else "cpu")
	macs, _ = profile(net, inputs=(input, query))
	total = sum([param.nelement() for param in net.parameters()])
	print('Macs = ' + str(macs/1000**3) + 'G')
	print('Params = ' + str(total/1e6) + 'M')

	from fvcore.nn import FlopCountAnalysis, parameter_count_table
	flops = FlopCountAnalysis(net, (input, query))
	print("FLOPs", flops.total()/1000**3)


