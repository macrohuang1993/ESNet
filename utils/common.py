import json, yaml
import logging

def load_loss_scheme(loss_config):

    with open(loss_config, 'r') as f:
        loss_json = yaml.safe_load(f)

    return loss_json

DEBUG =0
logger = logging.getLogger()

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
