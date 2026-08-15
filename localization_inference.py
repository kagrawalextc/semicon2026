import argparse
import cv2
import numpy as np


def preprocess(img):
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    return cv2.Canny(
        img,
        40,
        120
    )


def localize(reference_path, search_path):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(
            "Cannot read reference image: " + reference_path
        )

    if search is None:
        raise FileNotFoundError(
            "Cannot read search image: " + search_path
        )

    reference_edge = preprocess(reference)
    search_edge = preprocess(search)

    best = None

    # Reference/search scale ratio used by the PS2 synthetic setup.
    for scale in np.linspace(0.075, 0.125, 21):

        h, w = reference_edge.shape

        nw = max(8, int(round(w * scale)))
        nh = max(8, int(round(h * scale)))

        if nw >= search_edge.shape[1]:
            continue

        if nh >= search_edge.shape[0]:
            continue

        template = cv2.resize(
            reference_edge,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

        result = cv2.matchTemplate(
            search_edge,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, location = cv2.minMaxLoc(result)

        x = location[0] + nw / 2.0
        y = location[1] + nh / 2.0

        candidate = (
            float(score),
            float(x),
            float(y)
        )

        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        raise RuntimeError(
            "No valid localization candidate was found."
        )

    _, x, y = best

    return int(round(x)), int(round(y))


def main():

    parser = argparse.ArgumentParser(
        description="DRIFT-SENSE standalone localization inference"
    )

    parser.add_argument(
        "reference_image"
    )

    parser.add_argument(
        "search_image"
    )

    args = parser.parse_args()

    x, y = localize(
        args.reference_image,
        args.search_image
    )

    print(f"({x}, {y})")


if __name__ == "__main__":
    main()
