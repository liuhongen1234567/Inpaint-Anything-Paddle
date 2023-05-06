import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator


def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

#     if kind == 'pix2pixhd_multidilated':
#         return MultiDilatedGlobalGenerator(**kwargs)
    
#     if kind == 'pix2pixhd_global':
#         return GlobalGenerator(**kwargs)
    if kind == 'ffc_resnet':
        # pass
        return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')

