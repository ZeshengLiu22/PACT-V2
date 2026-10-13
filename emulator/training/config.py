"""Load the existing YAML schema and validate implemented capabilities."""

import yaml


def load_config(path):
    """Read YAML and reject unsupported pipeline options before initialization."""
    with open(path) as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config):
    """Validate the existing schema's implemented model, loss, and training modes."""
    train = config["training"]
    if (
        config["normalization"] != {"forcing": "mean_std", "target": "mean_std"}
        or not config["loss"]["global_mse"]
    ):
        raise ValueError("Only mean/std normalization and Global MSE are implemented")
    if (
        config["loss"]["tail"]["weight"] != 0
        or config["loss"]["episode_peak"]["weight"] != 0
    ):
        raise NotImplementedError("Tail and Episode Peak losses await the Episode Audit")

    # Reserved names must fail explicitly rather than silently using SmokeModel.
    if config["model"]["name"] != "smoke":
        if config["model"]["name"] in ("base", "igarss27", "full_v1", "full_v2"):
            raise NotImplementedError(
                f"{config['model']['name']} architecture has not been implemented"
            )
        raise ValueError("Unknown model identifier")

    evaluation = config["evaluation"]
    if (
        evaluation["checkpoint_selection"] != "overall"
        or not evaluation["save_overall_checkpoint"]
    ):
        raise NotImplementedError(
            "This initial pipeline requires best_overall checkpoint selection"
        )
    if evaluation["save_predictions"]:
        raise NotImplementedError("Full-field prediction export is not implemented")

    if (
        train["optimizer"] != "adam"
        or train["scheduler"] != "cosine"
        or train["amp_dtype"] != "bf16"
    ):
        raise ValueError("Use Adam, cosine scheduling, and bf16 AMP")
    if (
        min(
            train["epochs"], train["batch_size"], train["grad_accum_steps"],
            evaluation["val_every_epochs"],
        ) < 1
        or train["warmup_epochs"] < 0
        or train["warmup_epochs"] >= train["epochs"]
        or not 0 < train["warmup_start_factor"] <= 1
        or not 0 <= train["min_lr"] <= train["lr"]
        or train["lr"] <= 0
        or train["weight_decay"] < 0
        or train["max_grad_norm"] < 0
    ):
        raise ValueError("Invalid training/scheduler configuration")
