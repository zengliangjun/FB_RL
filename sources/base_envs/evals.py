import dataclasses

from base import configs


@dataclasses.dataclass
class BaseEvalConfig(configs.BaseConfig):
    name: str = ""


class EvalProxy():

    def __new__(self, cfg: BaseEvalConfig):
        return cfg.instantiate_from_config()

class Eval():
    def __init__(self, cfg: BaseEvalConfig):
        self.config = cfg


    def eval(self, *args, **kwargs) -> dict:
        pass
