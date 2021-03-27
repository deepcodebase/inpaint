import os
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm


mask_path = './masks'
mask_list = 'mask_list'

expected_area_min = 0.1
expected_area_max = 0.2

mask_path = '{}-{}-{}'.format(mask_path, expected_area_min, expected_area_max)
mask_list = '{}-{}-{}'.format(mask_list, expected_area_min, expected_area_max)

num = 1000

height = 512
width = 512
short_edge = min(height, width)

total_area = height * width
max_area = expected_area_max * total_area
min_area = expected_area_min * total_area

min_vertex = 10
max_vertex = 40

max_points = 5

start_area = 0.5

min_len = 0.1
max_len = 0.3

brush_size = int(0.02 * short_edge)

min_angle = -np.pi * 0.75
max_angle = np.pi * 0.75


def to_point(x, y):
    x = int(x * width)
    y = int(y * height)
    return (x, y)


def random_stroke():
    np.random.seed()
    n_vertex = np.random.randint(min_vertex, max_vertex+1)
    mask = np.ones((height, width), 'uint8') * 255

    points = np.random.randint(1, max_points+1)
    for i in range(points):
        angle = 0.
        last_x, last_y = np.random.uniform(start_area/2, 1-start_area/2, 2)
        for j in range(n_vertex // points):
            angle = np.pi - angle + np.random.uniform(min_angle, max_angle)
            if j % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.uniform(min_len, max_len)
            x = last_x + length * np.cos(angle)
            y = last_y + length * np.sin(angle)
            x, y = np.clip((x, y), 0, 1)

            cv2.line(
                mask, to_point(last_x, last_y), to_point(x, y), 0, brush_size)
            last_x, last_y = x, y
    return mask


def mask_ratio(mask):
    return 1 - np.mean(mask) / 255


def write_stroke(path):
    try:
        while True:
            mask = random_stroke()
            size = mask_ratio(mask)
            if expected_area_min <= size < expected_area_max:
                cv2.imwrite(path, mask)
                break
    except Exception as err:
        print(err, path)
        return path
    return True


def main():

    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)

    masks = [
        os.path.join(mask_path, 'mask_%d_%d_%06d.png') % (
            int(100*expected_area_min), int(100*expected_area_max), i)
        for i in range(num)]

    with mp.Pool() as p:
        with tqdm(total=len(masks)) as pbar:
            for i, res in tqdm(
                    enumerate(p.imap_unordered(write_stroke, masks))):
                if res is not True:
                    masks.remove(res)
                pbar.update()

    with open(mask_list, 'w') as f:
        f.write('\n'.join(masks))
    print('Now we have %d masks in: %s' % (len(masks), mask_path))
    print('List file location: %s' % mask_list)


if __name__ == '__main__':
    main()
