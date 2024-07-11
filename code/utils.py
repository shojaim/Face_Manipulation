import cv2
import numpy as np
from triangulation import scale_part, shift_chin
from skimage.transform import pyramid_gaussian, pyramid_expand
from skimage.util import img_as_float

dragging = False
selected_part = None
x_init, y_init = 0, 0
landmarks = []

def set_landmarks(new_landmarks):
    global landmarks
    landmarks = new_landmarks

def detect_face_part(x, y, landmarks):
    parts = {
        'eyes': list(range(36, 42)) + list(range(42, 48)),  # Combine both eyes
        'eyebrows': list(range(17, 22)) + list(range(22, 27)),  # Combine both eyebrows
        'nose': range(27, 36),
        'mouth': range(48, 68),
        'chin': list(range(5, 12)),  # Adjusted chin landmarks
    }

    for part_name, indices in parts.items():
        part_points = np.array([landmarks[i] for i in indices])
        if cv2.pointPolygonTest(part_points, (x, y), False) >= 0:
            return indices, part_name

    return None, None

def mouse_callback(event, x, y, flags, param):
    global dragging, selected_part, selected_part_name, x_init, y_init
    image = param['image']
    output_image = param['output_image']
    landmarks = param['landmarks']
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_part, selected_part_name = detect_face_part(x, y, landmarks)
        if selected_part:
            dragging = True
            x_init, y_init = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            output_image[:] = image.copy()
            if selected_part_name == 'chin':
                rect_shifted, dst_cropped, mask = shift_chin(selected_part, y, y_init, output_image, landmarks)
            else:
                rect_scaled, dst_cropped, mask = scale_part(selected_part, x, x_init, output_image, landmarks)
            if selected_part_name == 'chin':
                if (y - y_init) > 0:
                    output_image = blend_with_surroundings_increase(output_image, rect_shifted, dst_cropped, mask)
                else:
                    output_image = blend_with_surroundings_decrease(output_image, rect_shifted, dst_cropped, mask)
            else:
                if (x - x_init) > 0:
                    output_image = blend_with_surroundings_increase(output_image, rect_scaled, dst_cropped, mask)
                else:
                    output_image = blend_with_surroundings_decrease(output_image, rect_scaled, dst_cropped, mask)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def pyramid_blend(image1, image2, mask):
    image1 = img_as_float(image1)
    image2 = img_as_float(image2)
    mask = img_as_float(mask)

    mask_pyramid = tuple(pyramid_gaussian(mask, max_layer=6))

    image1_pyramid = tuple(pyramid_gaussian(image1, max_layer=6))
    image2_pyramid = tuple(pyramid_gaussian(image2, max_layer=6))

    blended_pyramid = []
    for im1, im2, m in zip(image1_pyramid, image2_pyramid, mask_pyramid):
        min_shape = np.minimum(im1.shape, im2.shape)
        im1 = im1[:min_shape[0], :min_shape[1]]
        im2 = im2[:min_shape[0], :min_shape[1]]
        m = m[:min_shape[0], :min_shape[1]]
        blended_layer = im1 * m + im2 * (1 - m)
        blended_pyramid.append(blended_layer)

    blended = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        blended = pyramid_expand(blended)
        min_shape = np.minimum(blended.shape, blended_pyramid[i].shape)
        blended = blended[:min_shape[0], :min_shape[1]] * 0.5 + blended_pyramid[i][:min_shape[0], :min_shape[1]] * 0.5

    return (blended * 255).astype(np.uint8)

def blend_with_surroundings_increase(image, rect_shifted, dst_cropped, mask):
    blended = pyramid_blend(image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]], dst_cropped, mask)
    image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]] = blended
    return image

def blend_with_surroundings_decrease(image, rect_shifted, dst_cropped, mask):
    blended = pyramid_blend(image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]], dst_cropped, mask)
    image[rect_shifted[1]:rect_shifted[1] + rect_shifted[3], rect_shifted[0]:rect_shifted[0] + rect_shifted[2]] = blended
    return image
