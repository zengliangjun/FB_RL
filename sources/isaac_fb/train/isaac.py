from trainer.humenv import trainer
from trainer.humenv import configs
from fbutils.actor_post import ActorValueType

from isaac_fb.models import isaac
from base_envs import buffers

import torch
import time
from tqdm import tqdm

class IsaacTrainer(trainer.FBTrainer):

    config: configs.TrainerConfig

    expert_buffer: buffers.BufferBase
    rollout_buffer: buffers.Rollout
    z_buffer: buffers.Rollout

    calcute: isaac.IsaacCalcute

    def __init__(self, cfg: configs.TrainerConfig):
        super(IsaacTrainer, self).__init__(cfg)

    def _postinit_dump(self):
        path = f"{self.workdir}/config.json"
        env_config = self.config.env_config
        self.config.env_config = None
        self.config.save(path)
        self.config.env_config = env_config

    @torch.no_grad()
    def _train_collect_one_step(self, context_items):
        step = context_items['step']
        context_z = context_items['context_z']
        done = context_items['done']
        observations = context_items['observations']
        privileges = context_items['privileges']

        ## step 1
        step_count = self.env.timestep()[:, None]
        context_z = self._maybe_update_rollout_context(z=context_z, step_count=step_count)
        if step <= self.config.seed_steps:
            action = self.env.sample_action()
        else:
            # this works in inference mode
            device = self.agent.model.config.device

            state = self.agent.model.calcute.state_prepose(observations.to(device))
            state = state.reshape((state.shape[0], -1))

            action = self.agent.model.calcute.act(state=state, z_policy=context_z.to(device), type=ActorValueType.SAMPLE)
            action = action.detach().to(context_z.device)

        next_obs, rewards, next_dones, next_infos = self.env.step(action)

        ## update context
        context_items['context_z'] = context_z
        context_items['done'] = next_dones
        context_items['observations'] = next_infos['observations']
        context_items['privileges'] = next_infos['privileges']

        if done.shape[0] == torch.sum(done.float()):
            return

        indexes = ~done
        data = {
            "observations": observations[indexes],
            "privileges": privileges[indexes],
            "action": action[indexes],
            "z": context_z[indexes],
            "step_count": step_count[indexes],
            "next": {
                "observations": next_infos['observations'][:, -1][indexes],
                "privileges": next_infos['privileges'][indexes],
                "rewards": rewards[indexes][:, None],
                "terminated": next_infos["terminated"][indexes][:, None],
                "truncated": next_infos["truncated"][indexes][:, None]
            },
        }
        self.rollout_buffer.extend(data)

    def train(self):

        num_metrics_updates = 0

        start_time = time.time()
        fps_start_time = time.time()

        self.agent.model.train()

        ## collect context
        obs, extras = self.env.reset()
        done = torch.zeros((obs.shape[0]), dtype = torch.bool).to(obs.device)

        collect_context = {
                "observations": extras["observations"],
                "privileges": extras["privileges"],
                'done': done,
                "context_z": None
            }

        total_metrics = {}
        for step in tqdm(range(0, int(self.config.max_steps))):
            collect_context['step'] = step

            self._train_collect(collect_context)

            if step <= self.config.seed_steps:
                continue

            ## update
            num_metrics_updates += self._train_update(step, total_metrics)

            ## log
            infos = [step, start_time, fps_start_time, num_metrics_updates, total_metrics]
            reset_logs = self._train_logs(infos)
            if reset_logs:
                num_metrics_updates = 0
                total_metrics = {}
                fps_start_time = time.time()
            # ulogger.info(f"_train_logs")

            if 0 == step % self.config.eval_every_steps:
                self._train_eval(step)

            self._train_save([step])
