from base_envs import trainers
from trainer.dmc import configs
import time
from tqdm import tqdm

class FBTrainer(trainers.BaseTrainer):

    config: configs.TrainerConfig

    def __init__(self, cfg: configs.TrainerConfig):
        super(FBTrainer, self).__init__(cfg)

    def train(self):

        total_metrics = {}
        num_metrics_updates = 0

        start_time = time.time()
        fps_start_time = time.time()

        self.agent.model.train()

        for step in tqdm(range(0, int(self.config.max_steps))):
            '''
            if step % self.config.eval_every_steps == 0:
                self.eval(t)
            '''

            # torch.compiler.cudagraph_mark_step_begin()
            metrics = self.agent.update(self.replay_buffer, step)

            # we need to copy tensors returned by a cudagraph module
            self._train_update_metrics(total_metrics, metrics)
            num_metrics_updates += 1

            ## log
            infos = [step, start_time, fps_start_time, num_metrics_updates, total_metrics]
            reset_logs = self._train_logs(infos)
            if reset_logs:
                num_metrics_updates = 0
                total_metrics = {}
                fps_start_time = time.time()
            # ulogger.info(f"_train_logs")

            self._train_save([step])
