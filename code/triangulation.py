import numpy as np
import cv2
from scipy.spatial import Delaunay

def get_triangles(points):
    tri = Delaunay(points)
    return tri.simplices

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri[:3]), np.float32(dst_tri[:3]))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def scale_part(part, x, x_init, image, landmarks):
    center = np.mean([landmarks[idx] for idx in part], axis=0)
    scale_factor = 1.0 + ((x - x_init) * 0.003)  # Adjust the scaling sensitivity here

    part_points = np.array([landmarks[idx] for idx in part])
    scaled_points = center + scale_factor * (part_points - center)

    triangles = get_triangles(part_points)
    rect = cv2.boundingRect(np.float32([part_points]))
    rect_scaled = cv2.boundingRect(np.float32([scaled_points]))

    src_cropped = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    dst_cropped = np.zeros_like(src_cropped)

    for tri in triangles:
        tri_src = [(part_points[i][0] - rect[0], part_points[i][1] - rect[1]) for i in tri]
        tri_dst = [(scaled_points[i][0] - rect_scaled[0], scaled_points[i][1] - rect_scaled[1]) for i in tri]
        dst_cropped = apply_affine_transform(src_cropped, tri_src, tri_dst, (rect_scaled[2], rect_scaled[3]))

    mask = np.zeros((rect_scaled[3], rect_scaled[2], 3), dtype=np.float32)
    for tri in triangles:
        tri_dst = [(scaled_points[i][0] - rect_scaled[0], scaled_points[i][1] - rect_scaled[1]) for i in tri]
        cv2.fillConvexPoly(mask, np.int32(tri_dst), (1.0, 1.0, 1.0), 16, 0)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    image[rect_scaled[1]:rect_scaled[1] + rect_scaled[3], rect_scaled[0]:rect_scaled[0] + rect_scaled[2]] = \
        image[rect_scaled[1]:rect_scaled[1] + rect_scaled[3], rect_scaled[0]:rect_scaled[0] + rect_scaled[2]] * \
        (1 - mask) + dst_cropped * mask

    return rect_scaled, dst_cropped, mask

def shift_chin(part, y, y_init, image, landmarks):
    delta_y = (y - y_init) * 0.05  # Adjust the shift sensitivity here

    part_points = np.array([landmarks[idx] for idx in part])
    shifted_points = part_points.copy()
    for idx in range(len(part_points)):
        shifted_points[idx][1] += delta_y

    triangles = get_triangles(part_points)
    rect = cv2.boundingRect(np.float32([part_points]))
    rect_shifted = cv2.boundingRect(np.float32([shifted_points]))

    src_cropped = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    dst_cropped = np.zeros_like(src_cropped)

    for tri in triangles:
        tri_src = [(part_points[i][0] - rect[0], part_points[i][1] - rect[1]) for i in tri]
        tri_dst = [(shifted_points[i][0] - rect_shifted[0], shifted_points[i][1] - rect_shifted[1]) for i in tri]
        dst_cropped = apply_affine_transform(src_cropped, tri_src, tri_dst, (rect_shifted[2], rect_shifted[3]))

    mask = np.zeros((rect_shifted[3], rect_shifted[2], 3), dtype=np.float32)
    for tri in triangles:
        tri_dst = [(shifted_points[i][0] - rect_shifted[0], shifted_points[i][1] - rect_shifted[1]) for i in tri]
        cv2.fillConvexPoly(mask, np.int32(tri_dst), (1.0, 1.0, 1.0), 16, 0)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]] = \
        image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]] * \
        (1 - mask) + dst_cropped * mask

    return rect_shifted, dst_cropped, mask
