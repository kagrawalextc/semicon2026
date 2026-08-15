import argparse
import csv
from pathlib import Path
import cv2
import numpy as np


def make_structure(size, architecture, rng):
    img = np.zeros((size, size), dtype=np.uint8)

    if architecture == "DRAM":
        px = int(rng.integers(18, 30))
        py = int(rng.integers(18, 30))

        for x in range(5, size, px):
            cv2.line(img, (x, 0), (x, size - 1), 170, 2)

        for y in range(5, size, py):
            cv2.line(img, (0, y), (size - 1, y), 170, 2)

        for x in range(5, size, px):
            for y in range(5, size, py):
                cv2.circle(img, (x, y), 3, 255, -1)

    elif architecture == "FinFET":
        pitch = int(rng.integers(15, 25))

        for x in range(5, size, pitch):
            cv2.line(img, (x, 0), (x, size - 1), 190, 2)

        for _ in range(3):
            y = int(rng.integers(size // 4, 3 * size // 4))
            cv2.rectangle(img, (0, y - 2), (size - 1, y + 2), 255, -1)

    else:
        raise ValueError("architecture must be DRAM or FinFET")

    edges = cv2.Canny(img, 40, 120)
    img[edges > 0] = 255
    return img


def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-SENSE synthetic reference/search image generator"
    )
    parser.add_argument(
        "--architecture",
        required=True,
        choices=["DRAM", "FinFET"]
    )
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with (out / "ground_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow([
            "reference",
            "search",
            "x",
            "y",
            "architecture"
        ])

        for i in range(args.pairs):

            reference = make_structure(
                100,
                args.architecture,
                rng
            )

            search = make_structure(
                1000,
                args.architecture,
                rng
            )

            # Independent sensor noise.
            reference = np.clip(
                reference.astype(np.float32) +
                rng.normal(0, 5, reference.shape),
                0, 255
            ).astype(np.uint8)

            search = np.clip(
                search.astype(np.float32) +
                rng.normal(0, 12, search.shape),
                0, 255
            ).astype(np.uint8)

            # Independent transformation.
            angle = float(rng.uniform(-3, 3))
            scale = float(rng.uniform(0.90, 1.10))

            matrix = cv2.getRotationMatrix2D(
                (50, 50),
                angle,
                scale
            )

            patch = cv2.warpAffine(
                reference,
                matrix,
                (100, 100),
                borderMode=cv2.BORDER_REFLECT
            )

            # Target centre.
            x = int(rng.integers(100, 900))
            y = int(rng.integers(100, 900))

            x0 = x - 50
            y0 = y - 50

            search[
                y0:y0 + 100,
                x0:x0 + 100
            ] = patch

            # SEM-like edge brightening.
            edges = cv2.Canny(search, 40, 120)
            search[edges > 0] = np.maximum(
                search[edges > 0],
                220
            )

            # Blur variation.
            if rng.random() < 0.5:
                search = cv2.GaussianBlur(
                    search,
                    (3, 3),
                    0.6
                )

            ref_name = f"reference_{i:04d}.png"
            search_name = f"search_{i:04d}.png"

            cv2.imwrite(
                str(out / ref_name),
                reference
            )

            cv2.imwrite(
                str(out / search_name),
                search
            )

            writer.writerow([
                ref_name,
                search_name,
                x,
                y,
                args.architecture
            ])

    print("Generated", args.pairs, "image pairs")
    print("Architecture:", args.architecture)
    print("Output:", out)


if __name__ == "__main__":
    main()
