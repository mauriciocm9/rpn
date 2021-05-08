import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import skimage.color
import skimage.io
import skimage.transform
import warnings
import scipy
from lib.config import Config
from lib.data_utils import DataSequence
from lib.model import RPN
from lib import utils as ut
import cv2

TRAIN_PATH = 'truset/train'
VALIDATION_PATH = 'truset/validation'


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    return skimage.transform.resize(
        image, output_shape,
        order=order, mode=mode, cval=cval, clip=clip,
        preserve_range=preserve_range, anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma)

class NucleiConfig(Config):

    NAME = 'nuclei'

    # Data parameters
    IMAGE_SHAPE = (512, 512)
    ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ANCHORS_PER_IMAGE = 128
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # Learning parameters
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    BATCH_SIZE = 8
    EPOCHS = 10


class NucleiSequence(DataSequence):

    def __init__(self, path, config):
        super().__init__(config)

        # Get the path to the data
        self.path = path

        # Image IDs are the folder names in this dataset
        self.image_ids = next(os.walk(self.path+"/fakes"))[2]
        np.random.shuffle(self.image_ids)

        # Store the configuration class
        self.config = config

        # Generate the anchors
        self.anchors = ut.generate_anchors(self.config.ANCHOR_SCALES,
                                           self.config.ANCHOR_RATIOS,
                                           ut.backbone_shapes(self.config.IMAGE_SHAPE, self.config.BACKBONE_STRIDES),
                                           self.config.BACKBONE_STRIDES,
                                           self.config.ANCHOR_STRIDE)

    def __len__(self):
        return int(len(self.image_ids) / self.config.BATCH_SIZE)

    def __getitem__(self, idx):
        # Choose the image ID's to be loaded into the batch
        image_ids = self.image_ids[idx * self.config.BATCH_SIZE: (idx + 1) * self.config.BATCH_SIZE]

        # Only RGB images - todo: fix this
        image_batch = np.zeros(((self.config.BATCH_SIZE, ) + self.config.IMAGE_SHAPE + (3,)))
        rpn_match_batch = np.zeros((self.config.BATCH_SIZE, self.anchors.shape[0], 1))
        rpn_bbox_batch = np.zeros((self.config.BATCH_SIZE, self.config.TRAIN_ANCHORS_PER_IMAGE, 4))

        # Load the batches
        for batch_idx in range(self.config.BATCH_SIZE):

            # Load the image and
            image, scale, padding = self.load_image(image_ids[batch_idx])
            bboxes = self.get_bboxes(image_ids[batch_idx], scale, padding)

            # Trim bboxes
            if bboxes.shape[0] > self.config.MAX_GT_INSTANCES:
                bboxes = bboxes[:self.config.MAX_GT_INSTANCES]

            # Generate the ground truth RPN targets to learn from
            rpn_match, rpn_bbox = ut.rpn_targets(self.anchors, bboxes, self.config)

            # Update the batch variables
            image_batch[batch_idx] = self.preprocess_image(image)
            rpn_match_batch[batch_idx] = np.expand_dims(rpn_match, axis=1)
            rpn_bbox_batch[batch_idx] = rpn_bbox

        # Store the inputs in a list form
        inputs = [image_batch, rpn_match_batch, rpn_bbox_batch]

        return inputs, []

    def load_image(self, _id):
        filename = os.path.join(self.path, "fakes", _id)
        image = cv2.imread(filename)
        max_dim = self.config.IMAGE_SHAPE[0]
        min_dim = self.config.IMAGE_SHAPE[0]

        # image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))

        # Does it exceed max dim?
        if max_dim:
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = resize(image, (round(h * scale), round(w * scale)),
                            preserve_range=True)

        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB).astype("float32")
        return image, scale, padding

    def get_bboxes(self, _id, scale, padding):
        # Get the filenames for all of the nuclei masks
        filename = os.path.join(self.path, "masks", _id)
        mask = cv2.imread(filename)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)

        bboxes = []
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts[0]:
            (x, y, w, h) = cv2.boundingRect(cnt)
            bboxes.append([y, x, y + h, x + w])

        if len(bboxes) == 0:
            bboxes.append([0,0,0,0])

        return np.array(bboxes)

    def preprocess_image(self, image):
        # Subtract the mean
        preprocessed_image = image.astype("float32") - self.config.MEAN_PIXEL
        return preprocessed_image


def main():

    # Configuration
    config = NucleiConfig()

    # Dataset
    dataset = {"train": NucleiSequence(TRAIN_PATH, config), "validation": NucleiSequence(VALIDATION_PATH, config)}

    # Region proposal network
    rpn = RPN(config)
    rpn.train(dataset)


if __name__ == '__main__':
    main()