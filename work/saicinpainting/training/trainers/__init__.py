import logging
import paddle
# from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
from saicinpainting.training.modules import make_generator

# def get_training_model_class(kind):
#     if kind == 'default':
#         return DefaultInpaintingTrainingModule

#     raise ValueError(f'Unknown trainer module {kind}')


# def make_training_model(config):
#     kind = config.training_model.kind
#     kwargs = dict(config.training_model)
#     kwargs.pop('kind')
#     kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

#     logging.info(f'Make training model {kind}')

#     cls = get_training_model_class(kind)
#     return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=True):

      generator = make_generator(train_config, **train_config.generator)
#     model: paddle.nn.Layer = make_training_model(train_config)
      # state = paddle.load('/home/aistudio/data/data211468/paddle_lanm.pdparams')
      # generator.set_state_dict(state)
      # generator.eval()

#     model.on_load_checkpoint(state)
#     return model
      return   generator

