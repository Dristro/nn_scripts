import torch

__all__ = ["save_model"]

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               format: str = "model",
               ):
    """
    Saves the model to the specified path with the name provided

    Args:
        model       - the model to save
        target_dir  - where to save the model (specific path, not relative)
        model_name  - name of the saved file
        format      - save format (weights, model, etc)
                      (model only as of now)

    Returns:
        No return, but saves the model to the given directory
    """

    from pathlib import Path
    # Create a target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True,
                          exist_ok = True)
    # Dave the model to the dir
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj = model.state_dict(),
               f = model_save_path)
    print(f"Model saved")


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None):
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance
    and saves to a specific log_dir.

    log_dir (default) - ./runs/timestamp/experiment_name/model_name/extra

    Args:
        experiment_name - name of the experiment
        model_name      - name of the model
        extra           - other stuff for you to add in the path

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter() - Instance of a writer that saves to log_dir
    """
    from datetime import datetime
    import os
    from torch.utils.tensorboard.writer import SummaryWriter

    timestamp = datetime.now().strftime("%d-%m-%Y")
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    print(f"[INFO] Created SummaryWriter, saves to: \033[1m{log_dir}\033[0m")
    return SummaryWriter(log_dir=log_dir)