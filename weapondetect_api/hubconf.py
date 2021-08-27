"""YOLOv5 PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""
import torch


def _create(name, verbose=True, device='cpu'):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 pytorch model
    """
    from pathlib import Path
    from models.experimental import attempt_load
    from utils.general import check_requirements, set_logging
    from utils.torch_utils import select_device

    file = Path(__file__).absolute()
    check_requirements(requirements=file.parent / 'requirements.txt', exclude=('tensorboard', 'thop', 'opencv-python'))
    set_logging(verbose=verbose)

    save_dir = Path('') if str(name).endswith('.pt') else file.parent
    path = (save_dir / name).with_suffix('.pt')  # checkpoint path
    try:
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
        print(f"Running on {device}")
        model = attempt_load(path, map_location=device)  # download/load FP32 model
        model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        print(e)
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pt', verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, verbose=verbose, device=None)

