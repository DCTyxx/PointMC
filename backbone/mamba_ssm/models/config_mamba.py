from dataclasses import dataclass, field

from utils.misc import ObjDict


@dataclass
class MambaConfig(dict):
    d_model: int = 256
    d_intermediate: int = 0
    n_layer: int = 1
    ssm_cfg: ObjDict = field(default_factory=ObjDict)
    attn_cfg: ObjDict = field(default_factory=ObjDict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    scan_method: str = ''

    @classmethod
    def default(cls, d_model: int = 256, n_layer: int = 1):
        return MambaConfig(
            d_model=d_model,
            d_intermediate=0,
            n_layer=n_layer,
            ssm_cfg=ObjDict(layer="Mamba2"),
            attn_cfg=ObjDict(num_heads=4),
            rms_norm=True,
            residual_in_fp32=True,
            scan_method='',
        )

if __name__ == '__main__':
    c = MambaConfig()
    print(c)
