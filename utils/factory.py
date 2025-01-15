from models.coil import COIL
from models.der import DER
from models.ewc import EWC
from models.finetune import Finetune
from models.foster import FOSTER
from models.gem import GEM
from models.icarl import iCaRL
from models.lwf import LwF
from models.replay import Replay
from models.bic import BiC
from models.podnet import PODNet
from models.rmm import RMM_FOSTER, RMM_iCaRL
from models.wa import WA
from models.myicarl import MyiCaRL
from models.myfinetune import MyFinetune
from models.mylwf import MyLwF
from models.myewc import MyEWC
from models.AID import AID
from models.AIDplus import AIDplus
from models.AID_time_guided import AIDTimeGuided
from models.AID_multiproto import AIDMultiProto
from models.AID_multiproto_adaptive import AIDMultiProtoAdaptive
from models.AID_proto_gmm import AIDProtoGmm
def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "bic":
        return BiC(args)
    elif name == "podnet":
        return PODNet(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "wa":
        return WA(args)
    elif name == "der":
        return DER(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "replay":
        return Replay(args)
    elif name == "gem":
        return GEM(args)
    elif name == "coil":
        return COIL(args)
    elif name == "foster":
        return FOSTER(args)
    elif name == "rmm-icarl":
        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        return RMM_FOSTER(args)
    elif name == "myicarl":
        return MyiCaRL(args)
    elif name == "myfinetune":
        return MyFinetune(args)
    elif name == "mylwf":
        return MyLwF(args)
    elif name == "myewc":
        return MyEWC(args)
    elif name == "aid":
        return AID(args)
    elif name == "aidplus":
        return AIDplus(args)
    elif name == "aid_time_guided":
        return AIDTimeGuided(args)
    elif name == "aid_multiproto":
        return AIDMultiProto(args)
    elif name == "aid_multiproto_adaptive":
        return AIDMultiProtoAdaptive(args)
    elif name == "aid_proto_gmm":
        return AIDProtoGmm(args)
    elif name == "aid_ci":
        from models.AID_CI import AID_CI
        return AID_CI(args)
    elif name == "mmd":
        from models.MMD import MMD
        return MMD(args)
    elif name == "adaptive":
        from models.adaptive import Adaptive
        return Adaptive(args)
    elif name == "dmr":
        from models.DMR import DMR
        return DMR(args)
    else:
        assert 0
