from models.mdan import MDAN
from models.mdanc import MDANC
from models.mdancbam import MDANCBAM
from models.mdancl import MDANCL
from models.mdancnr import MDANCNR
from models.mdancns import MDANCNS
from models.mdancnrs import MDANCNRS


def get_model(model_name, upscale_factor):

    if model_name == "MDAN":
        return MDAN(upscale_factor)

    if model_name == "MDANC":
        return MDANC(upscale_factor)

    if model_name == "MDANCBAM":
        return MDANCBAM(upscale_factor)

    if model_name == "MDANCL":
        return MDANCL(upscale_factor)

    if model_name == "MDANCNR":
        return MDANCNR(upscale_factor)

    if model_name == "MDANCNS":
        return MDANCNS(upscale_factor)

    if model_name == "MDANCNRS":
        return MDANCNRS(upscale_factor)

    else:
        raise NotImplementedError
