from .crn.trn import TransformerCRN
from .shot_encoder.resnet import resnet50
from .audio_encoder.LGSS_audionet import audnet

def get_shot_encoder(cfg):
    name = cfg.MODEL.visual.shot_encoder.name
    shot_encoder_args = cfg.MODEL.visual.shot_encoder[name]
    if name == "resnet":
        depth = shot_encoder_args["depth"]
        if depth == 50:
            shot_encoder = resnet50(
                pretrained=shot_encoder_args["use_imagenet_pretrained"],
                **shot_encoder_args["params"],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return shot_encoder

def get_audio_encoder(cfg):
    name = cfg.MODEL.audio.audio_encoder.name
    audio_encoder_args = cfg.MODEL.audio.audio_encoder.params[name]
    if name == "LGSS_audionet":
        audio_encoder = audnet(audio_encoder_args)
    else:
        raise NotImplementedError
    return audio_encoder


def get_contextual_relation_network(cfg, input_mode):
    crn = None
    if input_mode == 'visual':
        if cfg.MODEL.visual.contextual_relation_network.enabled:
            name = cfg.MODEL.visual.contextual_relation_network.name
            crn_args = cfg.MODEL.visual.contextual_relation_network.params[name]
            if name == "vis_cat":
                sampling_name = cfg.LOSS.sampling_method.name
                crn_args["neighbor_size"] = (
                    2 * cfg.LOSS.sampling_method.params[sampling_name]["neighbor_size"]
                )
                crn = TransformerCRN(crn_args)
            else:
                raise NotImplementedError
    elif input_mode == 'audio':
        if cfg.MODEL.audio.contextual_relation_network.enabled:
            name = cfg.MODEL.audio.contextual_relation_network.name
            crn_args = cfg.MODEL.audio.contextual_relation_network.params[name]
            if name == "aud_cat":
                sampling_name = cfg.LOSS.sampling_method.name
                crn_args["neighbor_size"] = (
                    2 * cfg.LOSS.sampling_method.params[sampling_name]["neighbor_size"]
                )
                crn = TransformerCRN(crn_args)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    return crn


__all__ = ["get_shot_encoder", "get_contextual_relation_network", "get_audio_encoder"]
