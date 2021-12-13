# _*_ coding : utf-8 _*_
# @Time : 2021/12/11 15:58
# @Author : wxs
# @File : endnconder
# @Project :
from torch import nn


# 编码器
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    # 输入一个 X，得到一个状态
    def forward(self, X, *args):
        raise NotImplementedError


# 解码器
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    # 编码器的输出转换成什么形式
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    # X：额外的输入，state：维护的状态
    def forward(self, X, state):
        raise NotImplementedError


# 合并编码器和解码器
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # 从编码器得到输出
        enc_outputs = self.encoder(enc_X, *args)
        # 将编码器的输出放入解码器，得到解码器的状态
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 将解码器的输入和解码器的状态放入解码器，得到输出
        return self.decoder(dec_X, dec_state)