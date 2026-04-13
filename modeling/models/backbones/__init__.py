import importlib


__factory = {
    'HRNet': ('modeling.models.backbones.hrnet', 'HRNet'),
    'ResNet': ('modeling.models.backbones.resnet', 'ResNet'),
    'RegNet': ('modeling.models.backbones.regnet', 'RegNet'),
    'ResNeXt': ('modeling.models.backbones.resnext', 'ResNeXt'),
    'STDCNet': ('modeling.models.backbones.stdcnet', 'STDCNet'),
    'DLAWrapper': ('modeling.models.backbones.dla', 'DLAWrapper'),
    'ShuffleNet': ('modeling.models.backbones.shufflenet', 'ShuffleNet'),
    'SECOND': ('modeling.models.backbones.second', 'SECOND'),
    'EfficientNet': ('modeling.models.backbones.efficientnet', 'EfficientNet'),
    'Transformer': ('modeling.models.backbones.transformer', 'Transformer'),
    'SwapTransformer': ('modeling.models.backbones.swap_transformer', 'SwapTransformer'),
    'SwinTransformer': ('modeling.models.backbones.swin_transformer', 'SwinTransformer'),
}


def names():
    return sorted(__factory.keys())


def create(name=None, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    module_name, class_name = __factory[name]
    module = importlib.import_module(module_name)
    factory = getattr(module, class_name)
    return factory(*args, **kwargs)
