import numpy as np

from numba import jit, njit, cuda

import time

@jit(nopython=True)
def test_kernel(img: np.ndarray) -> np.ndarray:
    return np.fft.fft2(img)

@cuda.jit()
def test_fft(img: np.ndarray) -> np.ndarray:
    return np.fft.fft2(img)


from cv2ext import IterableVideo


for fid, frame in IterableVideo("video.mp4"):
    print(fid, frame.shape)

    
    test_kernel(frame)
    test_fft(frame)
