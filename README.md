# DRIFT-SENSE

## AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

I4C Hackathon 2026 — Problem Statement 2 — Applied Materials Track

## Overview

DRIFT-SENSE locates a reference semiconductor pattern inside a larger
search image and returns the predicted center coordinate `(x, y)`.

## Final Localization Method

### Risk-Tiered Catastrophic-Safe Arbitration

The final system combines a primary localization estimate with a recovery
estimate. Candidate confidence, agreement, and displacement are evaluated
before accepting a recovery.

Larger displacement candidates are subjected to stricter acceptance rules.
This reduces unsafe jumps caused by repetitive semiconductor structures.

The verified configuration is stored in:

models/final_localizer_v6.json

## Repository

- dataset_generator.py
- localization_inference.py
- requirements.txt
- models/final_localizer_v6.json
- results/
- docs/

## Installation

    pip install -r requirements.txt

## Dataset Generation

Example:

    python dataset_generator.py --architecture DRAM --pairs 30 --output_dir generated_dram

FinFET:

    python dataset_generator.py --architecture FinFET --pairs 30 --output_dir generated_finfet

The generator creates actual reference/search image pairs and
ground_truth.csv containing the target centre coordinates.

## Inference

Run:

    python localization_inference.py reference.png search.png

The program prints one predicted coordinate:

    (x, y)

## Verified Benchmark

360 records were evaluated in the verified project benchmark.

Baseline mean error: 1.741681 px

Final mean error: 0.398950 px

Baseline accuracy within 1 px: 97.2222%

Final accuracy within 1 px: 97.5000%

Catastrophic baseline: 2

Catastrophic final: 1

New catastrophic cases: 0

Safety gate: PASSED

## Reproducibility

The repository contains standalone dataset generation, standalone inference,
the final configuration, verified benchmark results, dependencies, and references.

## References

See docs/references.md.
