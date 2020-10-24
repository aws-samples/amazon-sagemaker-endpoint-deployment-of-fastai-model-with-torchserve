import base64
import io
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)


class DIYSegmentation:
    """
    DIYSegmentation handler class.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        model_dir = properties.get("model_dir")

        manifest = ctx.manifest
        logger.error(manifest)
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        logger.debug(model_pt_path)

        from model import DynamicUnetDIY

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = DynamicUnetDIY()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file {0} loaded successfully".format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
        Scales and normalizes a PIL image for an U-net model
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        image_transform = transforms.Compose(
            [
                # must be consistent with model training
                transforms.Resize((96, 128)),
                transforms.ToTensor(),
                # default statistics from imagenet
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image)).convert(
            "RGB"
        )  # in case of an alpha channel
        image = image_transform(image).unsqueeze_(0)
        return image

    def inference(self, img):
        """
        Predict the chip stack mask of an image using a trained deep learning model.
        """
        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)
        logging.debug(outputs.shape)
        return outputs

    def postprocess(self, inference_output):

        if torch.cuda.is_available():
            inference_output = inference_output[0].argmax(dim=0).cpu()
        else:
            inference_output = inference_output[0].argmax(dim=0)

        return [
            {
                "base64_prediction": base64.b64encode(
                    inference_output.numpy().astype(np.uint8)
                ).decode("utf-8")
            }
        ]


_service = DIYSegmentation()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
