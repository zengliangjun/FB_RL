import time
import numpy as np
from loguru import logger as ulogger
import collections
import numbers

from humenv import bench

from trainer.humenv import eval_configs, humenvbench
from base import models
from base_envs import buffers

#from nets import proxy_net
#from data import buffer

class RewardEvaluation:

    def __init__(self, cfg: eval_configs.RewardEvaluationConfig):
        self.config = cfg

        self.reward_eval = bench.RewardEvaluation(
            tasks=cfg.reward_tasks,
            env_kwargs=cfg.env_kwargs,
            num_contexts=1,
            num_envs=cfg.num_envs,
            num_episodes=cfg.num_episodes,
        )

    def eval(self, model: models.BaseModelCalcute,
                   buffer: buffers.Rollout,
                   logger = None,
                   step : int = 0) -> dict:

        if buffer.empty():
            return {}

        eval_agent = humenvbench.RewardWrapper(
            model=model,
            inference_dataset=buffer,
            num_samples_per_inference=self.config.num_inference_samples,
            inference_function="reward_wr_inference",
            max_workers=1,
            process_executor=False,
        )

        reward_metrics = {}

        start_t = time.time()
        ulogger.info(f"Reward started at {time.ctime(start_t)}")

        reward_metrics = self.reward_eval.run(agent=eval_agent)

        duration = time.time() - start_t
        ulogger.info(f"Reward eval time: {duration}")

        if logger is not None:
            logger_pad = 35

            m_dict = {}
            avg_return = []
            for task in reward_metrics.keys():
                m_dict[f"reward_{task}/return"] = np.mean(reward_metrics[task]["reward"])
                m_dict[f"reward_{task}/return#std"] = np.std(reward_metrics[task]["reward"])
                avg_return.append(reward_metrics[task]["reward"])
            m_dict["reward/return"] = np.mean(avg_return)
            m_dict["reward/return#std"] = np.std(avg_return)
            m_dict["reward/time"] = duration

            logger.log_metrics(m_dict, step, ty='eval')


            ulogger_buffer = f"\n{' step:':>{logger_pad}} {step}\n"
            for k in sorted(list(m_dict.keys())):
                # ulogger_buffer += f"\t\t\t {k}: {logger_dict[k]}\n"
                ulogger_buffer += f"{k:>{logger_pad}}: {m_dict[k]}\n"
            ulogger.info(ulogger_buffer)

        return reward_metrics

class TrackingEvaluation:

    def __init__(self, cfg: eval_configs.TrackingEvaluationConfig):
        self.config = cfg

        self.tracking_eval = bench.TrackingEvaluation(
            motions=cfg.motions,
            motion_base_path=cfg.motions_root,
            env_kwargs=cfg.env_kwargs,
            num_envs=cfg.num_envs,
        )

    def eval(self, model: models.BaseModelCalcute,
                   logger = None,
                   step : int = 0) -> dict:

        eval_agent = humenvbench.TrackingWrapper(model=model)

        start_t = time.time()
        ulogger.info(f"Tracking started at {time.ctime(start_t)}")

        tracking_metrics = self.tracking_eval.run(agent=eval_agent)

        duration = time.time() - start_t
        ulogger.info(f"Tracking eval time: {duration}")
        ##
        if logger is not None:
            logger_pad = 35

            aggregate, m_dict = collections.defaultdict(list), {}
            for _, metr in tracking_metrics.items():
                for k, v in metr.items():
                    if isinstance(v, numbers.Number):
                        aggregate[k].append(v)
            for k, v in aggregate.items():
                m_dict[f"track/{k}"] = np.mean(v)
                m_dict[f"track/{k}#std"] = np.std(v)
            m_dict[f"track/time"] = duration

            logger.log_metrics(m_dict, step, ty='eval')


            ulogger_buffer = f"\n{' step:':>{logger_pad}} {step}\n"
            for k in sorted(list(m_dict.keys())):
                # ulogger_buffer += f"\t\t\t {k}: {logger_dict[k]}\n"
                ulogger_buffer += f"{k:>{logger_pad}}: {m_dict[k]}\n"
            ulogger.info(ulogger_buffer)

        return tracking_metrics

