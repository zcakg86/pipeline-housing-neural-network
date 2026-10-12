#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly JAVA_APP="$REPOSITORY_ROOT/java-app/house-price-app"

cd "$REPOSITORY_ROOT"
mkdir -p .cache/matplotlib

# Keep Python libraries isolated from the base image. Re-running this script is
# safe after a requirements change or a Codespace rebuild.
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --requirement requirements.txt
.venv/bin/python -m pip install --no-deps --editable .

chmod +x "$JAVA_APP/mvnw" "$JAVA_APP/fetch_rentcast.zsh"

# Resolve Java dependencies during environment creation so the first Quarkus
# launch does not pause for Maven downloads.
(cd "$JAVA_APP" && ./mvnw -q -DskipTests dependency:go-offline)

# A Codespace receives committed files only. Report deployment artifacts that
# must be committed or restored from external artifact storage before startup.
required_artifacts=(
  "data/sales_2020_25_with_predictions.csv"
  "java-app/house-price-app/src/main/resources/model-artifacts/model.onnx"
  "java-app/house-price-app/src/main/resources/model-artifacts/model_metadata.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/scalers.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/lightgbm.onnx"
  "java-app/house-price-app/src/main/resources/model-artifacts/lightgbm_metadata.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/lightgbm_model.txt"
  "java-app/house-price-app/src/main/resources/model-artifacts/feature_contract.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/local_market_snapshot.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/h3_l8_neighbor_cells.json"
  "java-app/house-price-app/src/main/resources/model-artifacts/king_county_water.geojson"
  "java-app/house-price-app/src/main/resources/model-artifacts/fred_indicators.csv"
)
missing=()
for artifact in "${required_artifacts[@]}"; do
  [[ -f "$REPOSITORY_ROOT/$artifact" ]] || missing+=("$artifact")
done

if (( ${#missing[@]} > 0 )); then
  printf '\nWARNING: the Java app is missing required runtime artifacts:\n'
  printf '  - %s\n' "${missing[@]}"
  printf 'Commit them or restore a validated deployment bundle before running the app.\n'
fi

printf '\nCodespace setup complete.\n'
printf '  Run tests:  make test\n'
printf '  Start app:  make java-dev\n'
printf '  Open port:  8080 (private by default in GitHub Codespaces)\n'
