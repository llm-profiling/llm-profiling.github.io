import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    config_path: Optional[str] = None
    output_dir: str = "/tmp/lmtracer_exports"
    log_level: str = "INFO"
    enable_profiling: bool = True
    num_flush: Optional[int] = None
    dump_mode: str = "doorbell"
    
    @classmethod
    def from_env(cls) -> 'EnvConfig':
        return cls(
            config_path=cls._get_config_path(),
            output_dir=cls._get_output_dir(),
            log_level=cls._get_log_level(),
            enable_profiling=cls._get_enable_profiling(),
            num_flush=cls._get_num_flush(),
            dump_mode=cls._get_dump_mode(),
        )
    
    @staticmethod
    def _get_config_path() -> Optional[str]:
        return os.environ.get('LMTRACER_CONFIG')
    
    @staticmethod
    def _get_output_dir() -> str:
        return os.environ.get('LMTRACER_OUTPUT_DIR', '/tmp/lmtracer_exports')
    
    @staticmethod
    def _get_log_level() -> str:
        level = os.environ.get('LMTRACER_LOG_LEVEL', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level not in valid_levels:
            return 'INFO'
        return level
    
    @staticmethod
    def _get_enable_profiling() -> bool:
        value = os.environ.get('LMTRACER_ENABLE_PROFILING', '1').lower()
        return value in ('1', 'true', 'yes', 'on')
    
    @staticmethod
    def _get_num_flush() -> Optional[int]:
        value = os.environ.get('LMTRACER_NUM_FLUSH')
        if value is None:
            return None
        try:
            num = int(value)
            if num <= 0:
                return None
            return num
        except ValueError:
            return None
    
    @staticmethod
    def _get_dump_mode() -> str:
        mode = os.environ.get('LMTRACER_DUMP_MODE', 'doorbell').lower()
        valid_modes = ['doorbell', 'polling']
        if mode not in valid_modes:
            return 'doorbell'
        return mode
    
    def print_config(self):
        print("[lmtracer EnvConfig] Current configuration:")
        print(f"  Config Path: {self.config_path or 'Not set (using preset)'}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"  Log Level: {self.log_level}")
        print(f"  Profiling Enabled: {self.enable_profiling}")
        print(f"  Num Flush: {self.num_flush or 'Default (engine-specific)'}")
        print(f"  Dump Mode: {self.dump_mode}")


# Global environment configuration instance
_env_config: Optional[EnvConfig] = None


def get_env_config() -> EnvConfig:
    global _env_config
    if _env_config is None:
        _env_config = EnvConfig.from_env()
    return _env_config


def reload_env_config() -> EnvConfig:
    global _env_config
    _env_config = EnvConfig.from_env()
    return _env_config


def set_env_config(config: EnvConfig):
    global _env_config
    _env_config = config
