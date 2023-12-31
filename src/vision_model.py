# import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from utils import print_time

def read_img(img_path: str):
    # print(img_path, flush=True)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)

def is_vision_model(model_name, model_weight):
    if getattr(torchvision.models.segmentation, model_name, False) and \
        getattr(torchvision.models.segmentation, model_weight, False):
        return True
    elif getattr(torchvision.models.detection, model_name, False) and \
        getattr(torchvision.models.detection, model_weight, False):
        return True
    elif getattr(torchvision.models, model_name, False) and \
        getattr(torchvision.models, model_weight, False):
        return True
    else:
        return False

class VisionModel:
    # # https://github.com/netx-repo/PipeSwitch/blob/f321d399e501b79ad51da13074e2aecda36cb06a/pipeswitch/worker_common.py#L40
    # def insert_layer_level_sync(self, mod):
    #     def hook_terminate(mod, input, output):
    #         torch.cuda.synchronize()
    #         print("added sync")
    #     if len(list(mod.children())) == 0:
    #         mod.register_forward_hook(hook_terminate)
    #     else:
    #         for child in mod.children():
    #             self.insert_layer_level_sync(child)

    def __init__(self, model_name: str, model_weight: str, device_id: int) -> None:
        # torch.cuda.set_device(device_id)
        self.device_id = device_id
        torch.hub.set_dir("torch_cache/")

        if getattr(torchvision.models.segmentation, model_weight, False):
            # a model from torchvision.models.segmentation
            self.weights = getattr(torchvision.models.segmentation, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.segmentation, model_name)
        elif getattr(torchvision.models.detection, model_weight, False):
            # a model from torchvision.models.detection
            self.weights = getattr(torchvision.models.detection, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.detection, model_name)
        elif getattr(torchvision.models, model_weight, False):
            # a model from torchvision.models or terminated with an exception
            self.weights = getattr(torchvision.models, model_weight).DEFAULT
            model_cls = getattr(torchvision.models, model_name)
        else:
            print(f"Unrecognized model weight {model_weight} and model name "
                  f"{model_name} in torchvision.", file=sys.stderr, flush=True)
            raise ValueError("Unrecognized model weight and model name in "
                             "torchvision.")
        with print_time(f'loading {model_name}', sys.stderr):
            self.model = model_cls(weights=self.weights).eval().cuda(self.device_id)

        # self.config = config

        # batch_size = self.config['batch_size'] if 'batch_size' in self.config else 1
        # # for miniconda3/envs/torch/lib/python3.9/site-packages/torchvision/models/detection/transform.py"
        # self.resize = config['resize']
        # if self.resize:
        #     os.environ['RESIZE'] = "true"
        # else:
        #     os.environ['RESIZE'] = "false"
        # sync_level = self.config.get('sync_level', "")
        # if sync_level == "layer":
        #     self.insert_layer_level_sync(self.model)
        # self.resize_size = tuple(config['resize_size'])
        # self.model_preprocess = self.weights.transforms()

    def infer(self, request):
        img_path = request['input_file_path']
        resize_size = tuple(request['resize_size'])
        batch_size = request['batch_size']
        # if self.resize:
        img = read_img(img_path).unsqueeze(0)
        img = F.interpolate(img, resize_size)
        # else:
        #     img = self.model_preprocess(read_img(img_path)).unsqueeze(0)
        img = torch.cat([img] * batch_size)
        img = img.cuda(self.device_id)
        res = self.model(img)
        torch.cuda.synchronize()
        return res
