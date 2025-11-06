
from dm_control import suite
from loguru import logger as ulogger
import gymnasium

import sys
import signal
import traceback

from envs.dmc import configs

class DMCWarp:

    def __init__(self, cfg: configs.DMCConfig):
        self.config = cfg

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        try:
            ulogger.info(f"Creating dmc with parallel environments")
            env = suite.load(
                domain_name=cfg.domain_name,
                task_name=cfg.task_name,
                environment_kwargs={"flat_observation": True},
            )

            self.envs = env
            ulogger.info("Successfully created dmc environments")
        except Exception as e:
            ulogger.error(f"Failed to create dmc environments: {e}")
            ulogger.error(traceback.format_exc())
            raise

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            ulogger.info(f"Received signal {signum}, shutting down environments gracefully")
            self.end()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
