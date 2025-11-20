from isaac_fb.updaters import actor

from isaac_fbv2.models import encode
from isaac_fbv2.updaters import configs

class ActorUpdater(actor.ActorUpdater):

    def __init__(self, cfg: configs.ActorConfig, model: encode.Model):
        super(ActorUpdater, self).__init__(cfg, model)
