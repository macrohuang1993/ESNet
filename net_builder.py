from __future__ import print_function

from networks.ESNet import ESNet
from networks.ESNet_M import ESNet_M
from networks.DispNetC import DispNetC
from networks.DispNetS import DispNetS
from networks.FADNet import FADNet
from networks.stackhourglass import PSMNet
from networks.GANet_deep import GANet
from utils.common import logger

SUPPORT_NETS = {
        'esnet': ESNet,
        'esnet_m':ESNet_M,
        'fadnet': FADNet,
        'dispnetc': DispNetC,
        'dispnets': DispNetS,
        'psmnet': PSMNet,
        'ganet':GANet,
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
