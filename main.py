import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


X_scale = [-1000, 0]
Y_scale = [10**-6, 10**-4]

# X_scale = [-1000, 0]
# Y_scale = [0, 7*10**21]


def quantize_colors(image, n_colors):
    (h, w) = image.shape[:2]
    image = image.reshape((h * w, 3))
    kmeans = KMeans(n_colors)
    kmeans.fit(image)
    new_colors = kmeans.cluster_centers_.astype("uint8")[kmeans.labels_]
    quantized_image = new_colors.reshape((h, w, 3))

    return quantized_image, kmeans.cluster_centers_.astype("uint8")


def is_colorful(bgr_color, saturation_threshold=25):
    color_np = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)
    saturation = hsv_color[0][0][1]

    return saturation > saturation_threshold


def get_contour_centroids(contours):
    # Dataset 1
    x_dashed = []
    y_dashed = []

    # Dataset 2
    x_line = []
    y_line = []

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x, center_y = 0, 0

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 100:
            # if perimeter is large, probably two points (sqauare and circle) were overlapped
            x_dashed.append(center_x)
            y_dashed.append(center_y)
            x_line.append(center_x)
            y_line.append(center_y)
        elif perimeter > 70:
            # square perimeter is larger than circle, hence it's a square point
            x_dashed.append(center_x)
            y_dashed.append(center_y)
        else:
            # circle point
            x_line.append(center_x)
            y_line.append(center_y)

    return x_dashed, y_dashed, x_line, y_line


def mask(image, color):
    white_bg = np.full(image.shape, 255, dtype=np.uint8)
    masked_image = cv2.bitwise_and(
        white_bg, white_bg, mask=cv2.inRange(image, color, color)
    )

    return masked_image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def erode(image):
    kernel = np.ones((10, 10), np.uint8)
    return cv2.erode(image, kernel)


def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def main(image_path, n_datasets):
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # X-axis scaling factor (linear)
    x_scaling = (X_scale[1] - X_scale[0]) / w

    # Y-axis scaling factor (log)
    y_scaling = np.log10(Y_scale[1] / Y_scale[0]) / h
    # y_scaling = (Y_scale[1] - Y_scale[0]) / h

    # Quantize the image colors to n_datasets + 3 (white+gray+black)
    quantized_image, centroids = quantize_colors(image, n_datasets + 3)

    # Get dataset colors
    dataset_colors = [
        tuple(bgr_color.tolist()) for bgr_color in centroids if is_colorful(bgr_color)
    ]

    for color in dataset_colors:
        masked_image = mask(quantized_image, color)
        grayscale_image = to_grayscale(masked_image)
        eroded_image = erode(grayscale_image)
        contours = find_contours(eroded_image)

        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        x_dashed, y_dashed, x_line, y_line = get_contour_centroids(contours)

        for point in list(zip(x_dashed, y_dashed)):
            cv2.circle(image, tuple(point), 1, (255, 0, 0))

        for point in list(zip(x_line, y_line)):
            cv2.circle(image, tuple(point), 1, (0, 255, 0))

        # cv2.imshow("Contours", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x_dashed = [[x * x_scaling + X_scale[0]] for x in x_dashed]
        y_dashed = [10 ** ((h - y) * y_scaling + np.log10(Y_scale[0])) for y in y_dashed]
        # y_dashed = [[(h - y) * y_scaling + Y_scale[0]] for y in y_dashed]
        x_line = [[x * x_scaling + X_scale[0]] for x in x_line]
        y_line = [10 ** ((h - y) * y_scaling + np.log10(Y_scale[0])) for y in y_line]
        # y_line = [[(h - y) * y_scaling + Y_scale[0]] for y in y_line]

        np.savez(
            f"current_hptm_{(','.join(map(str, color)))}.npz",
            np.array(x_dashed),
            np.array(y_dashed),
        )
        np.savez(
            f"current_data_{(','.join(map(str, color)))}.npz",
            np.array(x_dashed),
            np.array(y_dashed),
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(x_dashed, y_dashed, s=60, alpha=0.7, edgecolors="k")
        ax.scatter(x_line, y_line, s=60, alpha=0.7, edgecolors="k")
        ax.set_ylim(bottom=Y_scale[0], top=Y_scale[1])
        ax.set_yscale("log")
        plt.savefig(f"capacitance_{(','.join(map(str, color)))}.png")


main("graph_1.png", 5)
