from models.csrnet import CSRNet
from models.csrnet_context import CSRNetContext
from models.csrnet_pano import CSRNetPano
from models.mcnn import MCNN
from models.can import CAN
from models.panocsrnet import PanoCSRNet


def build_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "csrnet":
        return CSRNet(load_pretrained_vgg=True)
    elif model_name == "mcnn":
        return MCNN()
    elif model_name == "can":
        return CAN()
    elif model_name == "panocsrnet":
        return PanoCSRNet(out_h=64, load_pretrained_vgg=True)
    elif model_name == "csrnet_context":
        return CSRNetContext()
    elif model_name == "csrnet_pano":
        return CSRNetPano(out_h=64, load_pretrained_vgg=True)
    elif model_name == "csrnet_pano_circular_only":
        from models.csrnet_pano import CSRNetPanoCircularOnly
        return CSRNetPanoCircularOnly(out_h=64, load_pretrained_vgg=True)
    elif model_name == "csrnet_pano_latprior_only":
        from models.csrnet_pano import CSRNetPanoLatPriorOnly
        return CSRNetPanoLatPriorOnly(out_h=64, load_pretrained_vgg=True)
    # DACNet removed
    else:
        raise ValueError(f"Unknown model: {model_name}")