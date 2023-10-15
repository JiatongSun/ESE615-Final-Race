import cv2
import numpy as np
from scipy import ndimage
import csv
import yaml
import os
from shapely.geometry import Polygon, Point

# ratio < 1: close to inner
# ratio = 1: center line
# ratio > 1: close to outer
inner_outer_dist_ratio = 1.0

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

outer_safe_dist = 0
inner_safe_dist = 0


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

    # Use EXTERNAL contour in opencv to ensure the track is counter-clockwise
    curr_contours, curr_hierarchy = cv2.findContours(path_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if plot and not show_result([path_img], title="track"):
        exit(0)
    return np.squeeze(curr_contours[0])


def draw_lane(img, lane, color=(0, 0, 255)):
    h, w = img.shape[:2]
    for idx in range(len(lane) - 1):
        cv2.line(img, lane[idx], lane[idx + 1], color, 1)
    cv2.line(img, lane[-1], lane[0], color, 1)  # connect tail to head

    start = lane[0]
    vec = lane[1] - lane[0]
    direction = vec / np.linalg.norm(vec)
    end = start + direction * min([h, w]) * 0.1
    end = end.astype(int)
    cv2.arrowedLine(img, start, end, (238, 130, 238), 1, tipLength=0.2)


if __name__ == "__main__":
    # Get map name
    module = os.path.dirname(os.path.abspath(__file__))
    config_file = module + "/config/params.yaml"
    with open(config_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    input_map = parsed_yaml["map_name"]
    input_map_ext = parsed_yaml["map_img_ext"]
    clockwise = parsed_yaml["clockwise"]
    use_lanes = parsed_yaml["use_lanes"]
    lane_index = parsed_yaml["lane_index"]

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
    outer_safe_pixel = outer_safe_dist / scale
    inner_safe_pixel = inner_safe_dist / scale
    outer_dist -= outer_safe_pixel
    inner_dist -= inner_safe_pixel
    safe_from_wall = (outer_dist > 0) & (inner_dist > 0)
    ratio = np.abs(inner_dist) / (np.abs(outer_dist) + 1e-8)

    path = find_lane(inner_outer_dist_ratio, on_track, ratio, safe_from_wall)
    path = reorder_vertex(output_img, path)  # counter-clockwise by default
    if clockwise:
        path = np.flipud(path)

    # Plot final result
    res_img = cv2.cvtColor(~output_img, cv2.COLOR_GRAY2BGR)
    draw_lane(res_img, path)
    if not show_result([res_img], title="track"):
        exit(0)

    # Calculate distance to left border (inner) and right border (outer)
    path_x, path_y = path[:, 0], path[:, 1]
    pts = list(map(Point, zip(path_x.tolist(), path_y.tolist())))
    outer_dist = np.array([outer_poly.exterior.distance(pts[i]) for i in range(len(pts))])
    inner_dist = np.array([inner_poly.exterior.distance(pts[i]) for i in range(len(pts))])

    outer_dist -= 2 * WIDTH
    inner_dist -= 2 * WIDTH

    if clockwise:
        left_dist = outer_dist
        right_dist = inner_dist
    else:
        left_dist = inner_dist
        right_dist = outer_dist

    # Scale from pixel to meters, translate coordinates and flip y
    path_x = path_x * scale + offset_x
    path_y = (h - path_y) * scale + offset_y
    left_dist = left_dist * scale
    right_dist = right_dist * scale

    # Save result to csv file
    data = np.vstack((path_x, path_y, right_dist, left_dist)).T
    if use_lanes:
        csv_path = module + "/inputs/tracks/" + input_map + "_lane_" + str(lane_index) + ".csv"
    else:
        csv_path = module + "/inputs/tracks/" + input_map + ".csv"
    with open(csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["# x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        for line in data:
            csv_writer.writerow(line.tolist())
