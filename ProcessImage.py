import cv2

size = 64


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        border = longest - w
        left = border // 2
        right = border - left

    elif h < longest:
        border = longest - h
        top = border // 2
        bottom = border - top

    else:
        pass

    return top, bottom, left, right


def processImg(img, h=size, w=size):
    top, bottom, left, right = getPaddingSize(img)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT,
                             value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img