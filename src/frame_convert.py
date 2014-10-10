"""
    Modified file for frame_convert.py from python wrapper for OpenKinect.
    Functions adapted to transform image without using import cv.
    (not necessary for new OpenCV versions).
    Original file version at:
    https://github.com/OpenKinect/libfreenect/blob/master/wrappers/python/frame_convert.py
"""

import numpy as np

def video_cv(video):
    """Converts video into a BGR format for opencv
    This is abstracted out to allow for experimentation
    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB
    Returns:
        An opencv image who's datatype is 1 byte, 3 channel BGR
    """
    video = video[:, :, ::-1]  # RGB -> BGR
    return video.astype(np.uint8)

def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display
        This is abstracted to allow for experimentation with normalization
        Args:
            depth: A numpy array with 2 bytes per pixel
        Returns:
            A numpy array that has been processed whos datatype is unspecified
    """
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def pretty_depth_cv(depth):
    """Converts depth into a 'nicer' format for display
    This is abstracted to allow for experimentation with normalization
    Args:
        depth: A numpy array with 2 bytes per pixel
    Returns:
        An opencv image who's datatype is unspecified
    """
    video = video[:, :, ::-1]  # RGB -> BGR
    return video.astype(np.uint8)
