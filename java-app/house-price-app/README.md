# House Price App — Component Guide

A Quarkus application that loads property data, runs ONNX inference, and serves an interactive Leaflet.js map. This guide explains every source file, what it does, and how the components connect.

The map also serves optional static OSM overlays from
`src/main/resources/META-INF/resources/data/`: natural water, rail/light-rail
stations and routes, bus stops/stations and routes, categorized major roads,
and strict tagged-node motorway junctions. These GeoJSON files are fetched only
when their corresponding map checkbox is enabled. Rebuild the transport files
from the repository root with `make osm-transport`.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (index.html + Leaflet.js)                          │
│  ├── GET /api/sales/h3           → H3 hex layer             │
│  ├── GET /api/rentcast           → sales points layer       │
│  ├── GET /api/zillow             → Zillow listing points    │
│  ├── GET /api/redfin             → Redfin listing points    │
│  ├── GET /api/sales/community    → community polygons       │
│  ├── GET /api/performance        → community error chart    │
│  ├── GET /api/home-types         → filter dropdowns         │
│  ├── GET /api/stats              → sidebar stats            │
│  ├── GET /api/export/csv         → CSV download             │
│  ├── POST /api/fetch/rentcast    → live RentCast fetch      │
│  ├── POST /api/fetch/zillow      → live Zillow fetch        │
│  ├── POST /api/fetch/redfin      → live Redfin fetch        │
│  └── POST /api/agent/redfin/...  → AI price assessment      │
└──────────────────┬──────────────────────────────────────────┘
                   │ JAX-RS REST
┌──────────────────▼──────────────────────────────────────────┐
│  api/                                                        │
│  ├── MapResource.java     map data endpoints                │
│  ├── FetchResource.java   live API triggers + cooldown      │
│  ├── AgentResource.java   assessments + evaluation feedback │
│  ├── ExportResource.java  CSV download                      │
│  └── StartupLoader.java   data load on boot                 │
└──────────┬────────────────────────┬────────────────────────┘
           │                        │
┌──────────▼──────────┐  ┌──────────▼──────────────────────┐
│  service/           │  │  ingest/                         │
│  ├── PropertyStore  │  │  ├── DataIngestionService        │
│  └── H3Aggregation  │  │  ├── PropertyRecord              │
│      Service        │  │  ├── RentcastApiClient           │
└─────────────────────┘  │  └── ZillowApiClient             │
                         └──────────┬───────────────────────┘
                                    │
                         ┌──────────▼───────────────────────┐
                         │  model/                           │
                         │  ├── EmbeddingModel   (ONNX)     │
                         │  ├── LightGBMModel    (ONNX)     │
                         │  ├── GnnModel         (ONNX)     │
                         │  └── ModelArtifacts  (loaders)   │
                         └──────────────────────────────────┘
                         ┌──────────────────────────────────┐
                         │  watcher/                         │
                         │  └── FileWatcherService           │
                         └──────────────────────────────────┘
                         ┌──────────────────────────────────┐
                         │  agent/                           │
                         │  ├── RedfinPriceAssessmentAgent  │
                         │  ├── PropertyAssessmentEvidence  │
                         │  └── AgentEvaluationStore         │
                         └──────────────────────────────────┘
                         ┌──────────────────────────────────┐
                         │  cli/                             │
                         │  └── RentcastFetcher              │
                         └──────────────────────────────────┘
```

---

## Package by package

### `model/` — ONNX inference and artifact loading

#### `ModelArtifacts.java`
Loads all static model files from `src/main/resources/model-artifacts/` at startup (`@PostConstruct`).

| File | Contents |
|---|---|
| `scalers.json` | Mean and std dev for every scaled feature |
| `community_map.json` | H3 L8 hex → integer community ID |
| `h3_l9_neighbor_communities.json` | H3 hex → `int[7]` community indices (center + 6 neighbours) |
| `model_metadata.json` | Architecture dimensions, reference date, `n_communities` |

Key public methods:
- `scaleFeature(feature, value)` — `(value - mean) / scale`
- `inverseScaleLogPrice(scaled)` — recovers unscaled log price
- `getLogPriceScale()` — the log_price scaler std dev (used for uncertainty conversion)
- `lookupH3Neighbors(h3Index)` — returns `int[7]` neighbour community indices
- `lookupCommunity(h3Index)` — community ID string for display

#### `EmbeddingModel.java`
Wraps the ONNX session. Handles all feature engineering and inference.

The exported ONNX model has **5 inputs and 3 outputs**:

| Input | Shape | Description |
|---|---|---|
| `community_indices` | `[1, 7]` long | Center + 6 neighbour community indices |
| `property_features` | `[1, 4]` float | Scaled sqft, lot size, beds, and water proximity |
| `time_features` | `[1, 3]` float | Scaled time_trend, annual_sin, annual_cos |
| `market_features` | `[1, 2]` float | Scaled mortgage_rate, unemployment_rate |
| `local_market_features` | `[1, 7, 5]` float | Causal center-and-neighbor market summaries |

| Output | Shape | Description |
|---|---|---|
| `log_price_scaled` | `[1, 1]` | Scaled log price |
| `log_var_scaled` | `[1, 1]` | Log variance from uncertainty head |
| `cls_attention` | `[1, 4]` | CLS attention over community, property, time, market |

`predict()` returns a `PredictionResult` record:
```java
record PredictionResult(
    double  predictedPrice,      // $ price
    double  predictionStdPrice,  // $ std dev (delta method: price × std_log_price)
    double  predictionCvPct,     // 95% CI width as % of price (3.92 × std_log_price × 100)
    float[] clsAttention         // [community, property, time, market]
)
```

**Note on uncertainty calibration:** `predictionStdPrice` and `predictionCvPct` are only meaningful if the model was trained with `estimate_uncertainty=True` (NLL loss). With MSE-only training the uncertainty head exists but produces uncalibrated values. Retrain with NLL to get reliable confidence intervals.

#### `GnnModel.java`
Loads the independently trained monthly H3 GraphSAGE deployment. Python exports
the GraphSAGE **price head** as `gnn_price_head.onnx` and precomputes the dated
embedding for every H3 cell/month in `gnn_monthly_embeddings.bin.gz`. Java loads
that table once at startup and combines the selected cell-month embedding with:

| Input | Shape | Description |
|---|---|---|
| `gnn_embedding` | `[batch, 32]` | Precomputed two-layer GraphSAGE H3 representation |
| `property_features` | `[batch, 4]` | Scaled sqft, lot size, beds, water proximity |
| `time_features` | `[batch, 3]` | Scaled time trend, annual sine, annual cosine |
| `market_features` | `[batch, 2]` | Scaled mortgage and unemployment rates |

This avoids graph reconstruction and message passing during map loading or
individual predictions. Dates after the final exported month use its latest
causal embedding: this is leakage-safe but increasingly stale until retraining
and redeployment. An unknown H3 cell receives the model’s zero-embedding
fallback.

---

### `ingest/` — data parsing and transformation

#### `PropertyRecord.java`
Immutable record — the single unified representation of a property across all sources.

Key fields:
- **Identity**: `id`, `address`, `source` (`"sales"`, `"rentcast"`, `"zillow"`)
- **Location**: `lat`, `lng`, `h3Index`, `community`
- **Property**: `sqft`, `sqftLot`, `beds`, `baths`, `homeType`
- **Transaction/listing**: `saleDate` (prediction observation date for live listings),
  `salePrice` (listing price), `zestimate` (0 when unavailable), `listingUrl`,
  `description`, and `atAGlanceFacts`
- **Predictions**: `predictedPrice`, `pctError`, `lightgbmPredictedPrice`,
  `lightgbmPctError`, `gnnPredictedPrice`, `gnnPctError`
- **Uncertainty**: `predictionStdPrice`, `predictionCvPct`
- **Attention**: `clsAttention[4]` — CLS attention over [community, property, time, market]

#### `DataIngestionService.java`
Parses the historical CSV and canonical API-response JSON into `PropertyRecord` lists. JSON rows are normalized and compared with `PropertyStore`; unchanged `(source, id)` records skip inference, while new or changed rows are sent through one shared preparation pass and batched neural, LightGBM, and GNN inference.

| Method | Source |
|---|---|
| `ingestSalesCsv(file)` | `sales_2020_25.csv` — historical |
| `ingestRentcastJson(file)` | RentCast JSON |
| `ingestZillowJson(file)` | Zillow JSON |
| `ingestRedfinJson(file)` | Redfin JSON, including description and at-a-glance facts |

#### `RentcastApiClient.java` / `ZillowApiClient.java` / `RedfinApiClient.java`
HTTP clients for the respective APIs. Parameters come from `application.properties`. Return raw `List<Map<String, Object>>` for `DataIngestionService` to parse.

---

### `service/` — in-memory storage and aggregation

#### `PropertyStore.java`
In-memory store partitioned by source: `ConcurrentHashMap<source, ConcurrentHashMap<id, record>>`. Thread-safe — records can be upserted from the file watcher while the API serves reads. Deduplicates on `(source, id)`.

Key methods:
- `upsert(records)` — insert or overwrite
- `getSalesRecords()` — `"sales"` + `"rentcast"` combined
- `getZillowListings()` — `"zillow"` only

#### `H3AggregationService.java`
Groups filtered `PropertyRecord`s by `h3Index` and computes per-hex statistics.

`HexStats` record — one per hex:

| Field | Description |
|---|---|
| `avgSalePrice` | Mean actual sale price |
| `avgPredictedPrice` | Mean predicted price |
| `avgPctError` | Mean % prediction error |
| `avgSqft` | Mean living area |
| `numSales` | Record count |
| `avgPredStd` | Mean prediction std dev in $ |
| `avgPredCvPct` | Mean 95% CI width as % of price |
| `attnCommunity..attnMarket` | Mean CLS attention per token |
| `boundary` | GeoJSON polygon ring |

`toGeoJson()` sets `displayValue` to whichever field matches the `variable` query param. Supported variables include `pct_error`, `predicted_price_neural`, `predicted_price_lightgbm`, `sale_price`, `sqft`, `num_sales`, `pred_std`, `pred_cv_pct`, and the six `attn_*` values.

#### `SyntheticPredictionService.java`
Loads the packaged `synthetic_h3_l8_grid.json` once and uses batched ONNX
inference to predict the fixed 2,000 sqft, 4,000 sqft lot, 3-bed property in
every historical H3 level-8 cell. Results are cached for the three most recently
selected dates. It returns neural, LightGBM, and Spatial GNN predictions.
Neural intervals come from the uncertainty head; LightGBM uses
a 95% conformal interval calibrated on its chronological holdout.
The synthetic date slider runs monthly from today back to `2020-01-01`. This
demonstration endpoint intentionally reuses the current market snapshot for
older dates; production Zillow and sales inference retain the snapshot
look-forward guard.

---

### `api/` — REST endpoints

#### `MapResource.java`

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/rentcast` | Viewport-bounded GeoJSON — clustered at low zoom, detailed sales at high zoom | Sales points layer |
| `GET /api/zillow` | Viewport-bounded GeoJSON — clustered at low zoom, detailed listings at high zoom | Zillow layer |
| `GET /api/redfin` | Viewport-bounded GeoJSON including `description` and `atAGlanceFacts` | Redfin layer |
| `GET /api/sales/h3` | GeoJSON polygons — H3 hex grid | H3 sales layer |
| `GET /api/sales/community` | GeoJSON polygons — hexes by community | Community layer |
| `GET /api/synthetic?saleDate=YYYY-MM-DD` | GeoJSON H3 L8 polygons with neural, LightGBM, and GNN predictions, intervals, and attention | Synthetic layer |
| `GET /api/synthetic/meta` | Grid count and demonstration slider date bounds | Synthetic date control |
| `GET /api/synthetic/explanation?h3Index=...&saleDate=...` | Exact LightGBM TreeSHAP, neural group/feature Shapley, and 1,024-coalition GNN feature Shapley effects | Click-open synthetic feature panel |
| `GET /api/points/explanation?source=...&id=...` | The same three-model explanation for a stored Zillow, RentCast, or historical-sale point | Point-popup contribution button |
| `GET /api/performance` | JSON — quarterly error stats by community | Performance chart |
| `GET /api/home-types` | JSON — distinct home type strings | Property type dropdowns |
| `GET /api/stats` | JSON — summary counts and averages | Sidebar stats |

Both `/api/rentcast` and `/api/zillow` include `predictionStdPrice`, `predictionCvPct`, and all 6 `attn*` fields so the frontend can colour points by any variable.

The `/api/rentcast` endpoint accepts `source=rentcast|sales|all` to serve RentCast only, historical sales only, or both combined.
The point endpoints accept `west`, `south`, `east`, `north`, `zoom`, and `limit`.
Below zoom 13 they return H3 clusters. At zoom 13 or above they return individual
points inside the viewport, capped at 20,000, together with `matched`, `returned`,
`truncated`, and `clustered` response metadata.

Neural feature effects are projected onto their corresponding exact group totals, so they sum back to the neural prediction. The response also reports an approximate sampling standard error for each feature. Historical sales dated on or before the exported local-market snapshot cannot be reconstructed exactly in Java; their explanation response is explicitly marked as a current-snapshot demonstration and may differ from the row-specific prediction stored in the training CSV.

GNN effects are exact Shapley values for its price-head inputs: the dated
spatial H3 embedding, each property field, each time field, and each economic
field. The frozen GraphSAGE embedding remains one spatial effect because the
Java bundle deliberately does not contain the historical node snapshots or
graph-convolution layers needed to attribute its internal market fields.

Synthetic LightGBM explanations are calculated on demand so the main GeoJSON
response stays compact. ONNX remains the price-prediction runtime; the bundled
LightGBM text model is used only for exact contribution output, which is checked
against the ONNX log price. The native explanation binding requires OpenMP
(`brew install libomp` on macOS, or the distribution's `libgomp` package on
Linux).

#### `FetchResource.java`
Triggers live API fetches. All fetch endpoints require `?confirm=true`. RentCast calls use a 24-hour in-process cooldown plus the persistent shared 45-call budget; the cooldown for other sources is configured with `fetch.cooldown.hours`. Each response is atomically saved as a new timestamped JSON file. `FileWatcherService` is the sole ingestion owner, and the HTTP action waits for its completion before reporting success.

| Endpoint | Action |
|---|---|
| `POST /api/fetch/rentcast?confirm=true` | Calls RentCast, saves timestamped JSON, waits for watcher ingestion |
| `POST /api/fetch/zillow?confirm=true` | Calls Zillow, saves timestamped JSON, waits for watcher ingestion |
| `POST /api/fetch/redfin?confirm=true&allPages=true` | Calls Redfin, optionally follows pagination, and saves/ingests each successful full response before starting the next request |
| `GET /api/fetch/status` | Last fetch times and cooldown state |

Redfin responses are append-only at the file level: every successful ZIP/page
is written to its own `data/redfin/redfin_<run>_<sequence>_<zip>_page_<n>.json`
artifact. If a later request is denied or fails, the endpoint returns
`status: partial`; all earlier response files remain available and ingested.

#### `AgentResource.java`

Creates an evidence-grounded LangChain4j price-assessment agent for an individual
Redfin listing. The agent is backed by a Hugging Face Inference Provider model
through its OpenAI-compatible endpoint. It receives three bounded evidence
views: the selected listing (including `description` and `atAGlanceFacts`), all
three already-stored listing predictions plus their explanations/time inputs,
and deployed model-artifact H3 center/neighbor features with nearby Redfin
listings and their stored predictions. A compact `h3_market_surface.json` adds
the historical-sales-derived price level, regional premium, dispersion, decayed
sale support, and last-sale date for each covered H3 cell, plus nearest OSM city
and neighborhood labels. Those labels are nearest named place nodes, not legal
boundaries. The agent path never loads the historical sales CSV or OSM extract
and does not generate the full monthly property time series. Evidence is sent in
one model request rather than through provider-side tool calls; this avoids
GPT-OSS Harmony tool-parser incompatibilities and saves an extra inference round
trip.

| Endpoint | Action |
|---|---|
| `POST /api/agent/redfin/{listingId}/assessment` | Assess the selected listing; accepts an optional `{"question":"..."}` body |
| `POST /api/agent/assessments/{assessmentId}/feedback` | Append a 1–5 rating plus optional expected price and comments |
| `GET /api/agent/evaluations/summary` | Return assessment/feedback counts, average rating, and feedback records |

Every assessment is saved under `data/agent-evaluations/assessments/`. Feedback
is stored as a separate append-only record under
`data/agent-evaluations/feedback/`, so later feedback never replaces the
original assessment. These directories are ignored by Git because they may
contain listing data and user comments.

To enable assessments, save the Hugging Face token in
`java-app/house-price-app/.env` (do not commit it), then restart Quarkus:

```dotenv
HF_TOKEN=hf_your_token_here
# Optional; the default is shown below.
HF_MODEL=openai/gpt-oss-120b:fastest
```

The token needs permission to use Hugging Face Inference Providers. At map zoom
13 or higher, enable the Redfin layer, select a listing, and choose **Assess
price with AI**. Clicking that button sends the selected public listing plus
the locally derived prediction, explanation, H3 artifact, and nearby-listing evidence
to the configured Hugging Face provider. No evidence is sent before the user
requests an assessment.

#### `ExportResource.java`
`GET /api/export/csv?source=all|sales|rentcast|zillow|redfin` — streams all in-memory records as a CSV download.

#### `StartupLoader.java`
Runs once at boot. Loads the configured historical sales CSV and supported JSON files already present in the watch directories. Directories, non-JSON files, and archival files whose names contain `previous` are ignored.

---

### `watcher/` — filesystem monitoring

#### `FileWatcherService.java`

The watcher is closed and its executor is interrupted during Quarkus shutdown or
hot reload. Duplicate create events for the same normalized file path are ignored
within an application run, so one saved response has exactly one ingestion owner.
Watches `data/rentcast/`, `data/zillow/`, and `data/redfin/` using Java NIO
`WatchService` on a background daemon thread. When a new file appears, waits
for size stabilisation then ingests it. Skips files currently being written by
`FetchResource`.

---

### `cli/` — standalone utilities

#### `RentcastFetcher.java`
Charge-guarded standalone `main()` for fetching or explicitly paginating
RentCast without the app running. It reads geographic parameters from
`application.properties` and the API key from `.env`.

Safety behavior:

- No request is sent without `--confirm`.
- The default and minimum run is one call; `--max-calls` is explicit and has a
  hard per-run maximum of 5.
- A persistent 45-call budget is stored in
  `data/rentcast/.state/call_count`. Each call is reserved atomically before
  sending, so Ctrl-C or a crash cannot undercount API usage. The in-app
  `POST /api/fetch/rentcast` endpoint shares this same budget.
- Concurrent CLI processes are rejected with a file lock.
- There are no automatic HTTP retries.
- Every received page is immediately written atomically as
  `{"requestMetadata": ..., "properties": [...]}`, the format consumed by
  `DataIngestionService`. An aggregate file is also written after a successful
  run. In-app fetches also use unique timestamped response files and never
  overwrite an earlier successful response.

Example for exactly one call of up to 500 records:

```zsh
./fetch_rentcast.zsh --confirm
```

Stop at any time with Ctrl-C. To continue pagination later, use the
`nextOffset` printed by the preceding run, for example:

```zsh
./fetch_rentcast.zsh --confirm --offset=500
```

The wrapper defaults to `--limit=500 --max-calls=1 --offset=0`. It rejects
unknown arguments and still relies on the Java fetcher's persistent call
budget, locking, immediate response saving, and hard safety limits.

---

### `resources/` — static assets

#### `application.properties`

| Property | Default | Description |
|---|---|---|
| `data.sales.csv` | `../../data/sales_2020_25_with_predictions.csv` | Historical CSV loaded on startup |
| `watcher.rentcast.dir` | `data/rentcast` | Watched for new RentCast files |
| `watcher.zillow.dir` | `data/zillow` | Watched for new Zillow files |
| `watcher.redfin.dir` | `data/redfin` | Watched for new Redfin files |
| `fetch.cooldown.hours` | `0` | Minimum hours between non-RentCast live API calls (0 = disabled) |
| `fetch.rentcast.cooldown.hours` | `24` | Additional in-process cooldown for RentCast REST calls |
| `rentcast.months` | `4` | Completed-sale lookback window used in RentCast requests |
| `rentcast.timeout.seconds` | `180` | Maximum wait for one RentCast HTTP response; no automatic retry |
| `rentcast.api.key` | *(from env)* | RentCast API key |
| `zillow.api.key` | *(from env)* | Zillow/HasData API key |
| `carto.api.key` | *(from `CARTO_API_KEY`)* | Browser-visible CARTO Basemaps key; restrict it by website referrer in CARTO |
| `redfin.keywords` | Seattle/Eastside ZIP list | Comma-separated Redfin search keywords; one page is requested for each by default |
| `redfin.fetch.all.pages` | `false` | Follow Redfin response pagination by default |
| `redfin.max.pages` | `20` | Per-keyword billing safety cap for all-pages requests |
| `fetch.redfin.cooldown.hours` | `0` | Minimum time between Redfin fetches; set above zero for cost protection |
| `agent.huggingface.token` | *(from `HF_TOKEN`)* | Hugging Face token; leave unset to disable live assessments without preventing app startup |
| `agent.huggingface.base-url` | `https://router.huggingface.co/v1` | OpenAI-compatible Hugging Face router URL |
| `agent.huggingface.model` | `openai/gpt-oss-120b:fastest` | Inference Provider model and routing preference; override with `HF_MODEL` |
| `agent.timeout.seconds` | `180` | Maximum wait for an assessment response |
| `agent.evaluation.dir` | `data/agent-evaluations` | Append-only assessment and feedback artifacts |

#### `model-artifacts/`
Exported from the Python pipeline by `export_model_for_java.py`:

| File | Description |
|---|---|
| `model.onnx` | PyTorch model with 3 outputs: price, log_var, cls_attention |
| `scalers.json` | Feature scaler parameters |
| `community_map.json` | H3 hex → community ID |
| `h3_l9_neighbor_communities.json` | H3 hex → 7 community neighbour indices |
| `h3_market_surface.json` | Compact H3 historical-market aggregates plus nearest OSM city/neighborhood labels for agent evidence |
| `model_metadata.json` | Architecture dimensions, reference date, n_communities |
| `gnn_price_head.onnx` | GraphSAGE price head after precomputed spatial encoding |
| `gnn_monthly_embeddings.bin.gz` | Dated H3 cell embeddings, loaded once at startup |
| `gnn_scalers.json` / `gnn_metadata.json` | GNN input scaling and monthly snapshot metadata |

#### `META-INF/resources/index.html`
The entire frontend in a single file.

**Map layers:**

| Layer | Toggle | Colour driven by |
|---|---|---|
| H3 Sales | `showSales` | Variable selector (any variable) |
| Sales Points | `showRentcast` | Variable selector; source: RentCast / Historical / Both |
| Zillow Listings | `showZillow` | Variable selector |
| Community | `showCommunity` | Fixed 10-colour palette by community ID |

**Variable selector** — controls fill colour for H3 hexes, sales points, and Zillow points simultaneously:

| Value | Colour scale | Description |
|---|---|---|
| `pct_error` | Diverging blue→grey→red | Prediction error %, anchored to data p10/p90 |
| `sale_price` | YlOrRd | Average sale price |
| `sqft` | Blues | Average living area |
| `num_sales` | YlOrRd | Count of sales |
| `pred_std` | Blues | Prediction std dev in $ |
| `pred_cv_pct` | Blues | 95% CI width as % of price |
| `attn_community..attn_market` | YlOrRd | Average CLS attention weight per token |

**Tooltips** show: address, prediction/sale date, sale/list price, Zillow Zestimate
when applicable, predicted price, error %, 95% CI (`±$X ±Y%`), sqft/beds/baths,
and all 6 CLS attention weights as percentages.

**Colour functions:**
- `errorColor(v, anchors)` — diverging blue/grey/red, data-driven p10/p90 anchors, sign-preserving
- `getColorFn(variable, values, anchors)` — routes to `errorColor` or a sequential palette
- `pointVariableValue(props, variable)` — extracts the right field from a point feature for any variable
- `computeErrorAnchors(values)` — derives p10/p90 from the loaded dataset each refresh

---

## Data flow on a request

```
Browser changes variable selector to "pred_cv_pct"
  → loadSales() + loadZillow() + loadRentcast() all fire

loadSales():
  GET /api/sales/h3?variable=pred_cv_pct&...
    H3AggregationService.aggregateSales()
      PropertyStore.getSalesRecords()
      group by h3Index
      avgPredCvPct = mean(r.predictionCvPct()) per group
    toGeoJson() → displayValue = hex.avgPredCvPct()
  ← GeoJSON with displayValue = avg 95% CI width %

loadRentcast():
  GET /api/rentcast?source=rentcast&...
    MapResource.salesPoints()
      PropertyStore.getBySource("rentcast")
      props.predictionCvPct = r.predictionCvPct()
  ← GeoJSON points with predictionCvPct per point

Both layers:
  values = features.map(f => pointVariableValue / displayValue)
  getColorFn("pred_cv_pct", values, null)
    → Blues palette, p5–p95 normalised
  each feature coloured by its individual predictionCvPct
```

---

## Retraining and re-exporting

After retraining the Python models, deploy one consistent bundle from the
repository root:
```bash
PYTHONPATH=src python3 deploy_models_for_java.py
```
This exports neural, LightGBM, and GNN artifacts into a versioned `outputs/deployment/<timestamp>`
directory, verifies native/ONNX parity and feature ordering, writes
`manifest.json` with artifact hashes, and atomically replaces the sole canonical
deployment at `src/main/resources/model-artifacts/`. It also refreshes the
row-causal LightGBM columns in the historical prediction CSV used by the map.
Restart the app afterward.

To stage only the latest Java LightGBM artifacts, run from the repository root:

```bash
PYTHONPATH=src python3 export_lightgbm_for_java.py
```

This verifies the exported ONNX predictions against the trained Joblib model,
and stages `lightgbm.onnx`, `lightgbm_metadata.json`, and
`lightgbm_model.txt`. Add `--update-historical-predictions` to refresh the map
CSV as well. It does not change the live Java bundle unless `--deploy` is
supplied; the combined deploy command above is preferred because it prevents
mixed neural/LightGBM versions.

To regenerate the packaged synthetic grid after the historical H3 coverage
changes, run:

```bash
python3 generate_synthetic_h3_grid.py
```

This writes every historical H3 level-8 index, centroid, and closed polygon
boundary to `data/synthetic_h3_l8_grid.json`, then copies it to
both Java artifact directories. The application reads this file directly and
does not reconstruct the grid at runtime.

### Refreshing local market features without retraining

After saving new RentCast JSON or CSV pulls under
`java-app/house-price-app/data/rentcast/`, run from the repository root:

```bash
PYTHONPATH=src python3 refresh_local_market_snapshot.py
```

The command maintains `data/local_market_sales_ledger.csv`, deduplicates
RentCast transactions against the historical sales, and uses only sales dated
strictly before today's snapshot date. It refreshes these artifacts in both
Java artifact directories:

- `local_market_snapshot.json`
- `h3_l8_neighbor_cells.json`
- `h3_l8_neighbor_communities.json`

It does not change `model.onnx`, model weights, or scalers. Restart the Java
application after a successful refresh. To validate a scheduled run without
writing files, use:

```bash
PYTHONPATH=src python3 refresh_local_market_snapshot.py --dry-run
```

Use `--as-of YYYY-MM-DD` for a reproducible input cutoff. Only transactions
before that date are considered. The produced snapshot becomes usable on the
day after its latest included sale; Java rejects prediction dates on or before
that sale because they would look forward.

For point-in-time RentCast evaluation, preserve this order:

1. Let Java score new transactions whose sale dates are after the snapshot's
   `latest_sale_date`.
2. Export or otherwise retain those predictions.
3. Run the Python refresh to incorporate their observed prices into the next
   snapshot.

The snapshot now includes dated recent cell sales, allowing Java to recompute
the rolling 365-day trend and recency for each later prediction date.

For example, on macOS the scheduler-safe wrapper can run every day at 03:00
after the new RentCast records have been scored:

```cron
0 3 * * * /Users/marie/Documents/kiro/neural-network-house-prices/neural-networks-house-prices/scripts/refresh_local_market_snapshot.sh
```

Install that line with `crontab -e`, or use the equivalent LaunchAgent. The
wrapper uses the same Python 3.11 installation as model training and works
independently of the scheduler's current directory.

**For calibrated uncertainty**, train with `estimate_uncertainty=True` in
`main_train.py`. This changes the main objective from ordinary MSE to
Gaussian NLL: `exp(-log_var) * squared_error + log_var`. The mean prediction is
still trained through a squared residual, and the local model also retains its
auxiliary global-head MSE and residual penalty, but the total loss is not pure
MSE. Without NLL training, `predictionStdPrice` and `predictionCvPct` will be
present but unreliable.

---

## Adding a new map variable

1. Compute the per-record value during inference in `EmbeddingModel.predict()` or derive it in aggregation
2. Add it to `PropertyRecord` and `HexStats`
3. Compute the aggregate in `H3AggregationService.aggregateSales()`
4. Add a case to the `switch` in `toGeoJson()` and write it into `props`
5. Add `predictionCvPct` to both `/api/rentcast` and `/api/zillow` response properties
6. Add `<option>` to `<select id="variable">` in `index.html`
7. Add a case to `pointVariableValue()` and `getColorFn()` / legend `fmt`
