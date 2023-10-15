import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import yaml
import os
import shutil
from shapely.geometry import Polygon, Point

USE_CONVEX_HULL = False
USE_CORNER_CUT = True

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

safe_dist = 2 * WIDTH

# speed calculation params
beta = 4.0
horizon = 100
gamma = 0.99

lane_colors = [(0, 0, 255),
               (0, 255, 255),
               (0, 255, 0),
               (255, 255, 0),
               (255, 0, 0),
               (255, 0, 255)]


def show_result(imgs, title):
    if not imgs:
        return False
    height, width = imgs[0].shape[:2]
    w_show = 800
    scale_percent = float(w_show / width)
    h_show = int(scale_percent * height)
    dim = (w_show, h_show)
    img_resizes = []
    for img in imgs:
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_resizes.append(img_resize)
    img_show = cv2.hconcat(img_resizes)
    cv2.imshow(title, img_show)

    print("Press Q to abort / other keys to proceed")
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        return False
    else:
        cv2.destroyAllWindows()
        return True


def find_lane(lane_ratio, inside_track, dist_ratio, safe_area):
    valid_indices = np.where(inside_track & (np.abs(dist_ratio - lane_ratio) < lane_ratio / 10) & safe_area)[0]
    lane_pts = [pts[idx] for idx in valid_indices]
    lane_x = [lane_pts[idx].x for idx in range(len(lane_pts))]
    lane_y = [lane_pts[idx].y for idx in range(len(lane_pts))]
    return np.vstack((lane_x, lane_y)).T.astype(int)


def reorder_vertex(image, lane, plot=False):
    path_img = np.zeros_like(image)
    for idx in range(len(lane)):
        cv2.circle(path_img, lane[idx], 1, (255, 255, 255), 1)
    curr_kernel = np.ones((3, 3), np.uint8)
    iter_cnt = 0
    while True:
        if iter_cnt > 10:
            print("Unable to reorder vertex")
            exit(0)
        curr_contours, curr_hierarchy = cv2.findContours(path_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(curr_contours) == 2 and curr_hierarchy[0][-1][-1] == 0:
            break
        path_img = cv2.dilate(path_img, curr_kernel, iterations=1)
        iter_cnt += 1
    path_img = cv2.ximgproc.thinning(path_img)
    curr_contours, curr_hierarchy = cv2.findContours(path_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if plot and not show_result([path_img], title="track"):
        exit(0)
    return np.squeeze(curr_contours[0])


def draw_lane(img, lane, color=(0, 0, 255), show_arrow=True):
    h, w = img.shape[:2]
    for idx in range(len(lane) - 1):
        cv2.line(img, lane[idx], lane[idx + 1], color, 1)
    cv2.line(img, lane[-1], lane[0], color, 1)  # connect tail to head

    if show_arrow:
        start = lane[0]
        vec = lane[1] - lane[0]
        direction = vec / np.linalg.norm(vec)
        end = start + direction * min([h, w]) * 0.1
        end = end.astype(int)
        cv2.arrowedLine(img, start, end, (238, 130, 238), 1, tipLength=0.2)


def chaikins_corner_cutting(coords, refinements=5):
    for refine_idx in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def transform_coords(path, height, s, tx, ty):
    new_path_x = path[:, 0] * s + tx
    new_path_y = (height - path[:, 1]) * s + ty

    return np.vstack((new_path_x, new_path_y)).T


def calc_yaw(path):
    n_pts = len(path)
    yaws = []
    for i in range(n_pts):
        curr_point = path[i % n_pts]
        prev_point = path[(i - 1) % n_pts, 0:2]
        yaw = math.atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
        yaws.append(yaw)
    return yaws


def calc_curvature(path):
    dx_dt = np.gradient(path[:, 0])
    dy_dt = np.gradient(path[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    return curvature


def calc_speed(curvature, v_min, v_max):

    coeff = np.exp(-beta * curvature)
    curr_coeff = coeff.copy()
    for t in range(horizon):
        next_coeff = gamma * np.hstack((curr_coeff[1:], curr_coeff[0]))
        coeff += next_coeff
        curr_coeff = next_coeff.copy()
    coeff_min = np.min(coeff)
    coeff_max = np.max(coeff)
    speed = (coeff - coeff_min) / (coeff_max - coeff_min) * (v_max - v_min) + v_min
    return speed


def save_csv(data, csv_name):
    with open(csv_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["x", "y", "psi", "kappa", "v"])
        for line in data:
            csv_writer.writerow(line.tolist())


if __name__ == "__main__":
    # Get map name
    module = os.path.dirname(os.path.abspath(__file__))
    config_file = module + "/config/params.yaml"
    with open(config_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    input_map = parsed_yaml["map_name"]
    input_map_ext = parsed_yaml["map_img_ext"]
    max_speed = parsed_yaml["max_speed"]
    min_speed = parsed_yaml["min_speed"]
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

    # Define ratio = (dist to inner bound) / (dist to outer bound)
    np.arange(num_lanes + 1)
    lane_ratios = np.arange(1, num_lanes + 1) / np.arange(num_lanes, 0, -1)

    # Read image
    img_path = module + "/maps/" + input_map + input_map_ext
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape[:2]

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
    outer_poly = Polygon(outer_bound)
    inner_poly = Polygon(inner_bound)

    valid_pts = []
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = X.flatten().tolist()
    Y = Y.flatten().tolist()
    pts = list(map(Point, zip(X, Y)))
    on_track = np.array([outer_poly.contains(pt) and not inner_poly.contains(pt) for pt in pts])
    outer_dist = np.array([outer_poly.exterior.distance(pts[i]) for i in range(len(pts))])
    inner_dist = np.array([inner_poly.exterior.distance(pts[i]) for i in range(len(pts))])
    safe_pixel = safe_dist / scale
    outer_dist -= safe_pixel
    inner_dist -= safe_pixel
    safe_from_wall = (outer_dist > 0) & (inner_dist > 0)
    ratio = np.abs(inner_dist) / (np.abs(outer_dist) + 1e-8)

    # Calculate each lane
    lanes = []
    for lane_idx in range(len(lane_ratios)):
        print("Calculating lane" + str(lane_idx))
        lane = find_lane(lane_ratios[lane_idx], on_track, ratio, safe_from_wall)
        lane = reorder_vertex(output_img, lane)
        if USE_CONVEX_HULL:
            lane = np.squeeze(cv2.convexHull(lane, False))
        if clockwise:
            lane = np.flipud(lane)
        lanes.append(lane)

    # Plot final result
    res_img = cv2.cvtColor(~output_img, cv2.COLOR_GRAY2BGR)
    for idx in range(len(lanes)):
        draw_lane(res_img, lanes[idx], color=lane_colors[idx])

    if not show_result([res_img], title="track"):
        exit(0)

    # Scale from pixel to meters, translate coordinates and flip y
    for idx in range(len(lanes)):
        new_lane = transform_coords(lanes[idx], h, scale, offset_x, offset_y)
        if USE_CORNER_CUT:
            new_lane = np.vstack((new_lane, new_lane[0]))
            new_lane = chaikins_corner_cutting(new_lane)
            new_lane = new_lane[:-1]
        lanes[idx] = new_lane

    # Plot real-world coordinates
    plt.figure(figsize=(10, 8))
    for idx in range(len(lanes)):
        plt.plot(lanes[idx][:, 0], lanes[idx][:, 1], 'o', color='black')
        plt.axis('equal')
    # plt.show()

    # Calculate yaw and speed
    trajs = [np.zeros((len(lane), 5)) for lane in lanes]
    for idx in range(len(lanes)):
        trajs[idx][:, 0:2] = lanes[idx]
        trajs[idx][:, 2] = calc_yaw(lanes[idx])
        trajs[idx][:, 3] = calc_curvature(lanes[idx])
        trajs[idx][:, 4] = calc_speed(trajs[idx][:, 3], min_speed, max_speed)
        trajs[idx] = np.round(trajs[idx], 4)

    # Save to file
    csv_folder = os.path.join(module, "outputs", input_map)
    if os.path.exists(csv_folder):
        shutil.rmtree(csv_folder, ignore_errors=True)
    os.mkdir(csv_folder)
    for idx in range(len(lanes)):
        csv_path = os.path.join(csv_folder, "lane_" + str(idx) + "_naive.csv")
        save_csv(trajs[idx], csv_path)
