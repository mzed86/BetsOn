"""Enable running as: python -m src.models --config configs/model_config.yaml"""

from src.models.train import cli

if __name__ == "__main__":
    cli()
