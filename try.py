import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import warnings
import scipy
from lib import utils as ut


IMAGE_SHAPE = (512, 512)

# original = img_to_array(load_img("", target_size=IMAGE_SHAPE))

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    # if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
    return skimage.transform.resize(
        image, output_shape,
        order=order, mode=mode, cval=cval, clip=clip,
        preserve_range=preserve_range, anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma)
    # else:
    # return skimage.transform.resize(
    #     image, output_shape,
    #     order=order, mode=mode, cval=cval, clip=clip,
    #     preserve_range=preserve_range)

def read_image(file_name, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    image = cv2.imread(file_name)

    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, scale, padding

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    print(scale)
    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                        preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), scale, padding

def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def get_bboxes(mask):
    bboxes = []

    # print(mask.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = mask.astype('uint8')

    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts[0]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        bboxes.append([y, x, y + h, x + w])
        # cv2.rectangle(mask2, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if len(bboxes) == 0:
        bboxes.append([0,0,0,0])

    # cv2.imwrite(_id+".jpg, mask2)
    # print(_id)
    # print("DSS", bboxes)
    return np.array(bboxes)

ANCHOR_SCALES = (16, 32, 64, 128)
ANCHOR_RATIOS = [0.5, 1, 2]
ANCHOR_STRIDE = 1
BACKBONE_STRIDES = [4, 8, 16, 32, 64]

class Config(object):
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    TRAIN_ANCHORS_PER_IMAGE = 64

config = Config()

filename = "DCR215d40710e9494484fec5a7966c531e6_fake.jpeg"

original, scale, padding = read_image("truset/validation/fakes/"+filename, min_dim=512, max_dim=512, min_scale=0, mode="square")
mask = cv2.imread("truset/validation/masks/"+filename)
mask = resize_mask(mask, scale, padding)
bboxes = get_bboxes(mask)


anchors = ut.generate_anchors(ANCHOR_SCALES, ANCHOR_RATIOS, ut.backbone_shapes(IMAGE_SHAPE, BACKBONE_STRIDES), BACKBONE_STRIDES, ANCHOR_STRIDE)
rpn_match, rpn_bbox = ut.rpn_targets(anchors, bboxes, config)
ut.visualize_training_anchors(anchors, rpn_match, mask)
