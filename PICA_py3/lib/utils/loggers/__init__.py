#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

from lib.utils.loggers.std_logger import STDLogger
from lib.utils.loggers.tfb_logger import TFBLogger

from lib.core.config import Config as cfg

_MODULE_ = 'logger'
cfg.register_module(_MODULE_, __name__)
cfg.register_class(_MODULE_, 'std', STDLogger)
cfg.register_class(_MODULE_, 'tfb', TFBLogger)

def register(name, obj):
    cfg.register_class(_MODULE_, name, obj)