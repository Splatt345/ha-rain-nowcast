#!/usr/bin/env bash
# Downloads the real RMI (Belgium) radar example data pysteps ships for its
# own tests/examples (2021-07-04, 5-min QPE composites). Non-commercial use
# only per RMI's terms (see the pysteps-data repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
git clone --depth 1 https://github.com/pySTEPS/pysteps-data.git data/pysteps-data
