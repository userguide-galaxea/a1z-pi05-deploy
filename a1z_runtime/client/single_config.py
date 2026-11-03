# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Normalize the public single interface to the pinned model/SDK schema."""
def normalize_single_config(cfg):
    cfg = dict(cfg)
    if cfg.get("topology") == "pc-direct-single":
        cfg["topology"] = "pc-direct-single-left"
        for public, internal in (("can_channel", "left_can"),
                                 ("camera_wrist", "camera_left_wrist")):
            if public in cfg:
                cfg[internal] = cfg.pop(public)
    return cfg
