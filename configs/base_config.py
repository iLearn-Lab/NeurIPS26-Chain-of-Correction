from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import os

@dataclass
class ModelConfig:
    name: str = "qwen2.5-vl-3b"
    path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_seq_len: int = 2048
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

@dataclass
class SFTConfig:
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 4
    max_steps: int = 10000
    warmup_steps: int = 1000
    grad_accumulation_steps: int = 2

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.max_steps = int(self.max_steps)

@dataclass
class RLConfig:
    batch_size: int = 2
    learning_rate: float = 1e-6
    max_steps: int = 5000
    ppo_epochs: int = 4
    clip_coeff: float = 0.2

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.max_steps = int(self.max_steps)

@dataclass
class TrainingConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0

@dataclass
class DistributedConfig:
    num_gpus: int = 8
    strategy: str = "fsdp"
    backend: str = "nccl"
    find_unused_parameters: bool = False
    sharding_strategy: Optional[str] = None
    backward_prefetch: Optional[str] = None
    cpu_offload: bool = False
    mixed_precision_policy: Optional[str] = None

@dataclass
class DataConfig:
    root_path: str = "./data"
    sft_split: str = "train"
    rl_split: str = "train"
    sft_max_samples: Optional[int] = 40000
    num_workers: int = 4
    prefetch_factor: int = 2

@dataclass
class ProductionConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        # Convert dict inputs to proper config objects
        self._convert_dict_to_configs()
        self._validate()

    def _convert_dict_to_configs(self):
        """Convert dictionary inputs to proper config objects."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)

        if isinstance(self.training, dict):
            self.training = self._create_training_config(self.training)

        if isinstance(self.distributed, dict):
            self.distributed = DistributedConfig(**self.distributed)

        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)

    def _create_training_config(self, training_data: Dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from dictionary data."""
        if 'sft' in training_data:
            sft_config = SFTConfig(**training_data['sft'])
        else:
            sft_config = SFTConfig()

        if 'rl' in training_data:
            rl_config = RLConfig(**training_data['rl'])
        else:
            rl_config = RLConfig()

        # Extract non-nested fields
        training_fields = {k: v for k, v in training_data.items() if k not in ['sft', 'rl']}
        return TrainingConfig(sft=sft_config, rl=rl_config, **training_fields)

    def _validate(self):
        """Validate configuration parameters."""
        # Batch size validation
        if self.training.sft.batch_size <= 0:
            raise ValueError("SFT batch_size must be positive")
        if self.training.rl.batch_size <= 0:
            raise ValueError("RL batch_size must be positive")

        # GPU validation
        if self.distributed.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

        # Learning rate validation
        if self.training.sft.learning_rate <= 0:
            raise ValueError("SFT learning_rate must be positive")
        if self.training.rl.learning_rate <= 0:
            raise ValueError("RL learning_rate must be positive")

        # Model path validation
        if not self.model.path:
            raise ValueError("model path cannot be empty")

        # Data path validation
        if not self.data.root_path:
            raise ValueError("data root_path cannot be empty")

        # Steps validation
        if self.training.sft.max_steps <= 0:
            raise ValueError("SFT max_steps must be positive")
        if self.training.rl.max_steps <= 0:
            raise ValueError("RL max_steps must be positive")

        # Warmup steps should be less than max steps
        if self.training.sft.warmup_steps >= self.training.sft.max_steps:
            raise ValueError("SFT warmup_steps must be less than max_steps")

    @classmethod
    def from_yaml(cls, path: str) -> 'ProductionConfig':
        """Load configuration from YAML file with proper error handling."""
        try:
            # Validate file exists and is readable
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configuration file not found: {path}")

            if not os.access(path, os.R_OK):
                raise PermissionError(f"Cannot read configuration file: {path}")

            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if data is None:
                raise ValueError(f"Configuration file is empty or invalid: {path}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {path}: {e}")
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {path}: {e}")

        # Create config with data using helper method
        config = cls()

        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'training' in data:
            config.training = config._create_training_config(data['training'])
        if 'distributed' in data:
            config.distributed = DistributedConfig(**data['distributed'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])

        return config