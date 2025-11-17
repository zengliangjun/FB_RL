from base_envs import trainers
from trainer.humenv import configs
from fbutils.actor_post import ActorValueType

from cpr.models import fb
from base_envs import buffers

import torch
import time
from tqdm import tqdm

class FBTrainer(trainers.BaseTrainer):

    config: configs.TrainerConfig

    expert_buffer: buffers.BufferBase
    rollout_buffer: buffers.Rollout
    z_buffer: buffers.Rollout

    calcute: fb.FBCalcute

    def __init__(self, cfg: configs.TrainerConfig):
        super(FBTrainer, self).__init__(cfg)

        ### fb_cpr agent setup z buffer
        self.calcute = self.agent.model.calcute
        setattr(self.agent, "z_buffer", self.z_buffer)

    @torch.no_grad()
    def _maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor):
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.config.update_z_every_env_steps == 0
            if self.config.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=step_count.device)
            else:
                new_z = self.calcute.sample_z(z.shape[0], device=step_count.device)

            z = torch.where(mask_reset_z, new_z, z)
        else:
            z = self.calcute.sample_z(step_count.shape[0], device=step_count.device)
        return z

    @torch.no_grad()
    def _train_collect_one_step(self, context_items):
        step = context_items['step']
        context_z = context_items['context_z']
        obs = context_items['obs']
        done = context_items['done']
        infos = context_items['infos']

        ## step 1
        step_count = self.env.timestep()
        context_z = self._maybe_update_rollout_context(z=context_z, step_count=step_count)
        if step <= self.config.seed_steps:
            action = self.env.sample_action()
        else:
            # this works in inference mode
            device = self.agent.model.config.device
            state = self.agent.model.calcute.state_prepose(obs.to(device))
            action = self.agent.model.calcute.act(state=state, z_policy=context_z.to(device), type=ActorValueType.SAMPLE)
            action = action.detach().to(context_z.device)

        next_obs, rewards, next_dones, next_infos = self.env.step(action)

        ## update context
        context_items['context_z'] = context_z
        context_items['obs'] = next_obs
        context_items['done'] = next_dones
        context_items['infos'] = next_infos

        if self.config.env_config.num_envs == torch.sum(done.float()):
            return

        indexes = ~done
        data = {
            "observation": obs[indexes],
            "action": action[indexes],
            "z": context_z[indexes],
            "step_count": step_count[indexes],
            #"qpos": infos["qpos"][indexes],
            #"qvel": infos["qvel"][indexes],
            "next": {
                "observation": next_obs[indexes],
                "terminated": next_infos["terminated"][indexes],
                "truncated": next_infos["truncated"][indexes],
                #"qpos": next_infos["qpos"][indexes],
                #"qvel": next_infos["qvel"][indexes],
            },
        }
        self.rollout_buffer.extend(data)


    def _train_collect(self, context_items):
        for _ in range(self.config.one_step_collect_iters):
            self._train_collect_one_step(context_items)

    def _train_update(self, step, total_metrics):
        replay_buffer = {
            "expert_slicer": self.expert_buffer,
            "train": self.rollout_buffer,
        }

        for _ in range(self.config.one_step_update_iters):
            metrics = self.agent.update(replay_buffer, step)
            self._train_update_metrics(total_metrics, metrics)

        return self.config.one_step_update_iters

    def _train_eval(self, step):
        if not hasattr(self.agent.model.calcute, "seq_length"):
            setattr(self.agent.model.calcute, "seq_length", self.agent.config.seq_length)

        self.agent.model.eval()

        with torch.no_grad():
            if hasattr(self, "eval_reward") and self.eval_reward is not None:
                reward_metrics = self.eval_reward.eval(model = self.agent.model.calcute,
                                buffer = self.rollout_buffer,
                                logger = self.logger,
                                step = step)

            if hasattr(self, "eval_tracking") and self.eval_tracking is not None:
                tracking_metrics = self.eval_tracking.eval(model = self.agent.model.calcute,
                                logger = self.logger,
                                step = step)

                if self.expert_buffer.config.prioritization:
                    items = self.expert_buffer.priorities(tracking_metrics)
                    self.expert_buffer.update_priorities(items)
                    self.env.update_priorities(items)

        self.agent.model.train()

    def train(self):

        num_metrics_updates = 0

        start_time = time.time()
        fps_start_time = time.time()

        self.agent.model.train()

        ## collect context
        obs, extras = self.env.reset()
        done = torch.zeros((obs.shape[0]), dtype = torch.bool).to(obs.device)

        collect_context = {
                'obs': obs,
                'done': done,
                'infos': extras,
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
