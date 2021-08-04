import torch.nn as nn
import torch
from lib import Config as cfg
from lib.networks import DefaultModel, Flatten, register
from lib.utils.loggers import STDLogger as logger


__all__ = ['YOPO']

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class Block(nn.Module):

  def __init__(self, inplanes, planes, stride=2, track_running_stats=None):
    super(Block, self).__init__()

    assert (track_running_stats is not None)

    self.conv = conv3x3(inplanes, planes, stride)
    self.bn = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
    self.elu = nn.ELU(inplace=True)
    self.pool = nn.AdaptiveAvgPool3d()
    self.stride = stride

  def forward(self, x):

    out = self.conv(x)
    out = self.bn(out)
    out = self.elu(out)
    out = self.pool(out)

    return out


class YOPO(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')

    def __init__(self, cin, cout, sobel, net_heads=None, pool_size=None):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(YOPO, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 2 if sobel else cin
        self.layer1 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer2 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer3 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer4 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer5 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer6 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer7 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer8 = Block(self.inplanes, 128, 2, track_running_stats=True)
        self.layer9 = torch.cat()
        self.layer10 = nn.Sequential(nn.Dropout(p=0.5),
                                    nn.Linear(128*8, 1024),
                                    nn.ELU(0.3),
                                    nn.BatchNorm3d(1024),
                                    nn.Linear(1024, 256),
                                    nn.ELU(0.3),
                                    nn.BatchNorm3d(256),
                                    nn.Linear(256, 64)
                                    )
        heads = [nn.Sequential(nn.Linear(64, head),
                 nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)
        m1 = self.layer1(x)
        m2 = self.layer2(x)
        m3 = self.layer3(x)
        m4 = self.layer4(x)
        m5 = self.layer5(x)
        m6 = self.layer6(x)
        m7 = self.layer7(x)
        m8 = self.layer8(x)
        m= self.layer9([m1,m2,m3,m4,m5,m6,m7,m8])
        return map(lambda head:head(m), self.heads)


register('YOPO', YOPO)
