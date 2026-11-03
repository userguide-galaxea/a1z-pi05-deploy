# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Adapt public single collection names to the pinned collector schema."""
from pathlib import Path
import sys,tempfile
import yaml

def normalize(cfg):
    cfg=dict(cfg);arm=dict(cfg['arm_cfg']);camera=dict(cfg['camera_cfg'])
    cfg.pop('hardware_checks', None)  # Launcher-only options, outside upstream's structured schema.
    for public,internal in (('leader_port','leader_port_left'),('follower_can','follower_can_left')):
        if public in arm:arm[internal]=arm.pop(public)
    names={'head':'cam_high','wrist':'cam_left_wrist'}
    camera['camera_names']=[names.get(n,n) for n in camera['camera_names']]
    camera['topics']={names.get(k,k):v for k,v in camera['topics'].items()}
    if 'resolutions' in camera:
        camera['resolutions']={names.get(k,k):v for k,v in camera['resolutions'].items()}
    cfg['arm_cfg']=arm;cfg['camera_cfg']=camera
    return cfg

if __name__=='__main__':
    cfg=normalize(yaml.safe_load(Path(sys.argv[1]).read_text()))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.yaml',prefix='a1z_collect_',delete=False) as f:
        yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True);print(f.name)
