from .common import np
def ensure_uint16(img):
    return np.clip(img, 0, 65535).astype(np.uint16)