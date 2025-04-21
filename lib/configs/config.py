import json
import logging
import pathlib

import yaml
from easydict import EasyDict as edict

config = edict()

# 1. data_dir
config.data_dir = ''
config.model_dir = ''
config.log_dir = ''
config.tb_dir = ''
config.out_dir = ''
config.dataset = ''

# 2. data related
config.data = edict()
config.data.train_ds = ''
config.data.val_ds = ''
config.data.class_num = 19
config.data.in_channels = 3
config.data.crop_size = [700, 700]
config.data.num_workers = 8
config.data.mean = [0.485, 0.456, 0.406]
config.data.std = [0.229, 0.224, 0.225]
config.data.anomaly_mix = True
config.data.mixup = True

# 3. model related
config.model = edict()
config.model.weight_path = None
config.model.backbone = 'WideResNet38'
config.model.trainable_params_name = '.' # by default, use all params
config.model.trainable_params_name_update = None

# mask2former config synchronizer
config.model.mask2anomaly = edict()
config.model.mask2anomaly.use_official_loss = False
config.model.mask2anomaly.use_official_optimizer = False
config.model.mask2anomaly.use_official_params = False
config.model.mask2anomaly.use_official_train_mode = False
config.model.mask2anomaly.replace_official_odd_loss_with_RCL = False # RCL: Relative Contrastive Loss
config.model.mask2anomaly.deep_supervision = False
config.model.mask2anomaly.odd_weight = 1.0
config.model.mask2anomaly.mask_loss_with_pixel_selection = True

# 4. training params
config.train = edict()
config.train.n_epochs = 100
config.train.train_batch = 32
config.train.valid_batch = 32
config.train.test_batch = 1

config.train.optimizer = 'Adam'
config.train.lr = 1e-2
config.train.lr_update = None
config.train.momentum = 0.9
config.train.weight_decay = 1e-4

config.train.warmup_epoch = -1

# 5. loss related
config.loss = edict()
config.loss.name = ''
# config.loss.params = {
#     'ce_weights': None,
#     'inoutaug_contras_margins_tri': None,
#     'mask2anomaly_loss_weight': None,
# }

# update method
def update_config(config_file, id=None):
    logger = logging.getLogger()

    def recursively_update(key, value, sub_config):
        if isinstance(value, dict):
            for k_, v_ in value.items():
                if k_ not in sub_config:
                    logger.warning(f"cfg.{key}.{k_} is not in default config but in experiment config, added anyway...")
                    sub_config[k_] = v_
                else:
                    sub_config[k_] = recursively_update(f"{key}.{k_}", v_, sub_config[k_])
            return sub_config
        else:
            return value

    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                config[k] = recursively_update(k, v, config[k])
            else:
                logger.warning(f"cfg.{k} is not in default config but in experiment config, added anyway...")
                config[k] = v

    if id is not None:
        pathlib.Path(f"ckpts/{id}").mkdir(parents=True, exist_ok=True)

        with open(f"ckpts/{id}/config.yaml", 'w') as f:
            yaml.dump(json.loads(json.dumps(config)), f)
