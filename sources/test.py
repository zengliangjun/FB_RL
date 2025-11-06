import os
import os.path as osp

dir_root = osp.abspath(osp.dirname(__file__))

import sys
sys.path.insert(0, dir_root)

from base import configs, models
import inspect


if __name__ == "__main__":

    config = models.BaseModelConfig(
        _target_ = "base.models:BaseModel",
        _network_names_ = []
    )

    assert isinstance(config, models.BaseModelConfig)
    print(isinstance(config, models.BaseModelConfig))

    model = models.ModelsProxy(config)
    print(model)
    print(inspect.getmodule(model).__package__)
    print(type(model).__name__)
    print(model.__class__.__module__)
    print(model.__class__.__name__)
