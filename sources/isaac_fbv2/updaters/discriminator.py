
from cpr.updaters.cpr import discriminator_updater


from isaac_fbv2.models import encode
from isaac_fbv2.updaters import configs

class DiscriminatorUpdater(discriminator_updater.DiscriminatorUpdater):

    def __init__(self, cfg: configs.DiscriminatorConfig, model: encode.Model):
        super(DiscriminatorUpdater, self).__init__(cfg, model)
