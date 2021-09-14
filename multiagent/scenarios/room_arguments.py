import argparse
import random

def get_room_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--room_num', type=int, default=1)
    parser.add_argument('--wall_num', type=int, default=7)
    parser.add_argument('--wall_centers', type=list, default=[])
    parser.add_argument('--wall_shapes', type=list, default=[])

    args = parser.parse_args()

    args.wall_num = args.room_num * 7
    for i in range(args.room_num):
        id = random.randint(0, 3)
        wall_centers, wall_shapes = get_room(id)
        args.wall_centers.extend(wall_centers)
        args.wall_shapes.extend(wall_shapes)

    return args


def get_room(room_id=0):
    wall_centers = []
    wall_shapes = []
    if room_id == 0:
        wall_centers = [[0, 0.45], [-0.45, 0], [0.45, 0], [-0.3, -0.45], [0.3, -0.45], [-0.15, -1], [0.15, -1]]
        wall_shapes = [[1, 0.1], [0.1, 0.8], [0.1, 0.8], [0.4, 0.1], [0.4, 0.1], [0.1, 1], [0.1, 1]]
    if room_id == 1:
        wall_centers = [[0, -0.45], [-0.45, 0], [0.45, 0], [-0.3, 0.45], [0.3, 0.45], [-0.15, 1], [0.15, 1]]
        wall_shapes = [[1, 0.1], [0.1, 0.8], [0.1, 0.8], [0.4, 0.1], [0.4, 0.1], [0.1, 1], [0.1, 1]]
    if room_id == 2:
        wall_centers = [[0.45, 0], [0, -0.45], [0, 0.45], [-0.45, -0.3], [-0.45, 0.3], [-1, -0.15], [-1, 0.15]]
        wall_shapes = [[0.1, 1], [0.8, 0.1], [0.8, 0.1], [0.1, 0.4], [0.1, 0.4], [1, 0.1], [1, 0.1]]
    if room_id == 3:
        wall_centers = [[-0.45, 0], [0, -0.45], [0, 0.45], [0.45, -0.3], [0.45, 0.3], [1, -0.15], [1, 0.15]]
        wall_shapes = [[0.1, 1], [0.8, 0.1], [0.8, 0.1], [0.1, 0.4], [0.1, 0.4], [1, 0.1], [1, 0.1]]
    return wall_centers, wall_shapes
