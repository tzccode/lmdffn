from models.resnet import AcResNet, AcResNet_Thin, AcResNet_Thin_Random
from models.resnet import Light, Light_v2, Light_v3
from models.imdn import IMDN
from models.rfdn import RFDN
from models.awsrn import MODEL
from models.awac import AWAC
from models.awdc import AWDC
from models.awnac import AWNAC
from models.awaci import AWACI
from models.awfn import AWFN
from models.awffn import AWFFN
from models.mdarn import MDARN
from models.mdarn_l import MDARN_L
from models.awmfan import AWMFAN
from models.mdan import MDAN
from models.mdanc import MDANC
from models.mdancbam import MDANCBAM
from models.mdancl import MDANCL
from models.mdancnr import MDANCNR
from models.mdancns import MDANCNS
from models.mdancnrs import MDANCNRS
from models.acnet import Net
from models.srgan import Generator
from models.srganPR import SRGANPR
from models.SRSPR import SRSPR
from models.ssrp import SSRP
from models.ssrpr import SSRPR
from models.tsrp import TSRP
from models.tsrpr import TSRPR
from models.thsrpr import THSRPR
from models.thsrp import THSRP
from models.tbsrp import TBSRP
from models.tbsrpr import TBSRPR
from models.thdsrp import THDSRP


def get_model(model_name, upscale_factor):
    if model_name == 'AcResNet':
        return AcResNet(upscale_factor)

    if model_name == 'AcResNet_Thin':
        return AcResNet_Thin(upscale_factor)

    if model_name == 'AcResNet_Thin_Random':
        return AcResNet_Thin_Random(upscale_factor)

    if model_name == 'Light':
        return Light(upscale_factor)

    if model_name == 'Light_v2':
        return Light_v2(upscale_factor)

    if model_name == 'Light_v2_255':
        return Light_v2(upscale_factor)

    if model_name == 'Light_v3':
        return Light_v3(upscale_factor)

    if model_name == 'IMDN':
        return IMDN(upscale_factor)

    if model_name == 'RFDN':
        return RFDN(upscale_factor)

    if model_name == 'AWSRN':
        return MODEL(upscale_factor)

    if model_name == 'AWAC':
        return AWAC(upscale_factor)

    if model_name == 'AWDC':
        return AWDC(upscale_factor)

    if model_name == 'AWNAC':
        return AWNAC(upscale_factor)

    if model_name == 'AWACI':
        return AWACI(upscale_factor)

    if model_name == 'AWFN':
        return AWFN(upscale_factor)

    if model_name == 'AWFFN':
        return AWFFN(upscale_factor)

    if model_name == "MDARN":
        return MDARN(upscale_factor)

    if model_name == "MDARN_L":
        return MDARN_L(upscale_factor)

    if model_name == "AWMFAN":
        return AWMFAN(upscale_factor)

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

    if model_name == "ACNet":
        return Net()

    if model_name == "SRGAN":
        return Generator()

    if model_name == "SRGANPR":
        return SRGANPR(upscale_factor)

    if model_name == "SRSPR":
        return SRSPR(upscale_factor)

    if model_name == "SSRP":
        return SSRP(upscale_factor)

    if model_name == "SSRPR":
        return SSRPR(upscale_factor)

    if model_name == "TSRP":
        return TSRP(upscale_factor)

    if model_name == "TSRPR":
        return TSRPR(upscale_factor)

    if model_name == "THSRP":
        return THSRP(upscale_factor)

    if model_name == "THSRPR":
        return THSRPR(upscale_factor)

    if model_name == "TBSRP":
        return TBSRP(upscale_factor)

    if model_name == "TBSRPR":
        return TBSRPR(upscale_factor)

    if model_name == "THDSRP":
        return THDSRP(upscale_factor)

    else:
        raise NotImplementedError
