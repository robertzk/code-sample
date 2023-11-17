from dataclasses import dataclass

@dataclass
class CheckpointModelConfig:
    checkpoint_dir: str
    checkpoint_interval: int # Checkpoint every n training steps
    