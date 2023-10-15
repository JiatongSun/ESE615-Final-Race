import cv2
import numpy as np
import csv
import yaml
import os
from shapely.geometry import Polygon, Point

from lane_generator import show_result, reorder_vertex, draw_lane, transform_coords

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

safe_dist = 2 * WIDTH

module = os.path.dirname(os.path.abspath(__file__))


def read_lane(map_name, lane_idx):
    csv_loc = os.path.join(module, "outputs", map_name, "lane_" + str(lane_idx) + "_naive.csv")
    waypoints = np.loadtxt(csv_loc, delimiter=",", skiprows=1)
    return waypoints[:, :2]


def calc_dist_bound(inner_arr, outer_arr):
    outer_poly = Polygon(outer_arr)
    inner_poly = Polygon(inner_arr)

    valid_pts = []
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = X.flatten().tolist()
    Y = Y.flatten().tolist()
    pts = list(map(Point, zip(X, Y)))
    on_track = np.array([outer_poly.contains(pt) and not inner_poly.contains(pt) for pt in pts])
    outer_dist = np.array([outer_poly.exterior.distance(pts[i]) for i in range(len(pts))])
    inner_dist = np.array([inner_poly.exterior.distance(pts[i]) for i in range(len(pts))])

    return inner_dist, outer_dist, on_track, pts


def calc_track(inner_arr, outer_arr, lane, clockwise):
    outer_poly = Polygon(outer_arr)
    inner_poly = Polygon(inner_arr)

    lane_x, lane_y = lane[:, 0], lane[:, 1]
    pts = list(map(Point, zip(lane_x.tolist(), lane_y.tolist())))
    track_outer_dist = np.array([outer_poly.exterior.distance(pts[i]) for i in range(len(pts))])
    track_inner_dist = np.array([inner_poly.exterior.distance(pts[i]) for i in range(len(pts))])

    if clockwise:
        left_dist = track_outer_dist
        right_dist = track_inner_dist
    else:
        left_dist = track_inner_dist
        right_dist = track_outer_dist

    return left_dist, right_dist


def inv_transform_coords(path, height, s, tx, ty):
    new_path_x = ((path[:, 0] - tx) / s).astype(int)
    new_path_y = (height - (path[:, 1] - ty) / s).astype(int)

    return np.vstack((new_path_x, new_path_y)).T


def save_csv(data, csv_name):
    with open(csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["# x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        for line in data:
            csv_writer.writerow(line.tolist())


if __name__ == "__main__":
    # Get map name
    config_file = module + "/config/params.yaml"
    with open(config_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    input_map = parsed_yaml["map_name"]
    input_map_ext = parsed_yaml["map_img_ext"]
    num_lanes = parsed_yaml["num_lanes"]
    clockwise = parsed_yaml["clockwise"]

    # Read map params
    yaml_file = module + "/maps/" + input_map + ".yaml"
    with open(yaml_file, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(0)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    # Read image
    img_path = "./maps/" + input_map + input_map_ext
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape[:2]

    # Read lane files
    lanes = []
    for i in range(num_lanes):
        lane = read_lane(input_map, i)
        lane = inv_transform_coords(lane, h, scale, offset_x, offset_y)
        _, idx = np.unique(lane, axis=0, return_index=True)
        lane = lane[np.sort(idx)]
        lanes.append(lane)

    # Flip black and white
    output_img = ~input_img

    # Convert to binary image
    ret, output_img = cv2.threshold(output_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # Find contours and only keep larger ones
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 70:
            cv2.fillPoly(output_img, pts=[contour], color=(0, 0, 0))

    # Dilate & Erode
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.dilate(output_img, kernel, iterations=1)
    output_img = cv2.ximgproc.thinning(output_img)

    # Show images
    if not show_result([input_img, output_img], title="input & output"):
        exit(0)

    # Separate outer bound and inner bound
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parents = hierarchy[0][:, 3]
    assert np.max(parents) >= 1  # at least 3 levels for a valid track

    node = np.argmax(parents)
    tree_indices = []
    while node != -1:
        tree_indices.append(node)
        node = parents[node]
    tree_indices.reverse()

    outer_bound = np.squeeze(contours[tree_indices[1]])
    inner_bound = np.squeeze(contours[tree_indices[2]])

    # Euclidean distance transform
    inner_dist, outer_dist, on_track, pts = calc_dist_bound(inner_bound, outer_bound)
    safe_pixel = safe_dist / scale
    outer_dist -= safe_pixel
    inner_dist -= safe_pixel

    safe_outer_indices = np.where(on_track & (np.abs(outer_dist) < 2))[0]
    safe_outer_pts = [pts[idx] for idx in safe_outer_indices]
    safe_outer_x = [safe_outer_pts[idx].x for idx in range(len(safe_outer_pts))]
    safe_outer_y = [safe_outer_pts[idx].y for idx in range(len(safe_outer_pts))]
    safe_outer_arr = np.vstack((safe_outer_x, safe_outer_y)).T.astype(int)
    safe_outer_arr = reorder_vertex(output_img, safe_outer_arr)

    safe_inner_indices = np.where(on_track & (np.abs(inner_dist) < 2))[0]
    safe_inner_pts = [pts[idx] for idx in safe_inner_indices]
    safe_inner_x = [safe_inner_pts[idx].x for idx in range(len(safe_inner_pts))]
    safe_inner_y = [safe_inner_pts[idx].y for idx in range(len(safe_inner_pts))]
    safe_inner_arr = np.vstack((safe_inner_x, safe_inner_y)).T.astype(int)
    safe_inner_arr = reorder_vertex(output_img, safe_inner_arr)

    # Calculate each track's left and right distance to border
    bounds = [safe_inner_arr]
    for i in range(num_lanes):
        bounds.append(lanes[i])
    bounds.append(safe_outer_arr)

    idx_horizon = int((num_lanes + 1) / 2)
    left_dists = []
    right_dists = []
    for i in range(1, len(bounds) - 1):
        # inner_idx = max(0, i - idx_horizon)
        # outer_idx = min(len(bounds) - 1, i + idx_horizon)
        inner_idx = 0
        outer_idx = len(bounds) - 1
        left_dist, right_dist = calc_track(bounds[inner_idx], bounds[outer_idx], lanes[i - 1], clockwise)
        left_dist = left_dist * scale
        right_dist = right_dist * scale
        left_dists.append(left_dist)
        right_dists.append(right_dist)

    # Plot final result
    safe_img = cv2.cvtColor(~output_img, cv2.COLOR_GRAY2BGR)
    draw_lane(safe_img, safe_outer_arr, color=(0, 255, 0), show_arrow=False)
    draw_lane(safe_img, safe_inner_arr, color=(0, 255, 0), show_arrow=False)
    for idx in range(num_lanes):
        draw_lane(safe_img, lanes[idx], color=(0, 0, 255))
    if not show_result([safe_img], title="track"):
        exit(0)

    # Scale from pixel to meters, translate coordinates and flip y
    for idx in range(len(lanes)):
        new_lane = transform_coords(lanes[idx], h, scale, offset_x, offset_y)
        lanes[idx] = new_lane

    # Save result to csv file
    csv_folder = os.path.join(module, "inputs", "tracks")
    for filename in os.listdir(csv_folder):
        prefix = input_map + "_lane_"
        if prefix in filename:
            csv_path = os.path.join(csv_folder, filename)
            os.remove(csv_path)

    for idx in range(len(lanes)):
        csv_path = os.path.join(csv_folder, input_map + "_lane_" + str(idx) + ".csv")
        data = np.vstack((lanes[idx].T, right_dists[idx], left_dists[idx])).T
        save_csv(data, csv_path)
