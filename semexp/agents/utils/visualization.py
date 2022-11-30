import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat


def init_vis_image(goal_name, legend, num_pf_maps=0, add_sem_seg=False):
    H, W = 655, 1165
    if add_sem_seg:
        W += 15 + 640
    if num_pf_maps > 0:
        W += (15 + 480) * num_pf_maps
    vis_image = np.ones((H, W, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations (Goal: {})".format(goal_name)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(
        vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
    )

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(
        vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
    )

    # draw outlines
    color = [100, 100, 100]
    # RGB
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    # Semantic map
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    # Semantic seg
    pf_init_x = 1150
    if add_sem_seg:
        pf_init_x += 640 + 15
        vis_image[49:530, 1164] = color
        vis_image[49:530, 1805] = color
        vis_image[49, 1164:1805] = color
        vis_image[530, 1164:1805] = color

    # PF maps
    for i in range(num_pf_maps):
        start_x = pf_init_x + 15 * (i + 1) + 480 * i
        start_y = 50
        end_x = start_x + 480
        end_y = start_y + 480
        vis_image[start_y - 1 : end_y, start_x - 1] = color
        vis_image[start_y - 1 : end_y, end_x] = color
        vis_image[start_y - 1, start_x - 1 : end_x] = color
        vis_image[end_y, start_x - 1 : end_x] = color

    # draw legend
    lx, ly, _ = legend.shape
    vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend

    return vis_image
