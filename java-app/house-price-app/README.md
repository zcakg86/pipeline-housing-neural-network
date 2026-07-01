# House Price App — Component Guide

A Quarkus application that loads property data, runs ONNX inference, and serves an interactive Leaflet.js map. This guide explains every source file, what it does, and how the components connect.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (index.html + Leaflet.js)                          │
│  ├── GET /api/sales/h3           → H3 hex layer             │
│  ├── GET /api/sales/points       → heatmap layer            │
│  ├── GET /api/rentcast           → sales points layer       │
│  ├── GET /api/zillow             → Zillow listing points    │
│  ├── GET /api/sales/community    → community polygons       │
│  ├── GET /api/performance        → community error chart    │
│  ├── GET /api/home-types         → filter dropdowns         │
│  ├── GET /api/stats              → sidebar stats            │
│  ├── GET /api/export/csv         → CSV download             │
│  ├── POST /api/fetch/rentcast    → live RentCast fetch      │
│  └── POST /api/fetch/zillow      → live Zillow fetch        │
└──────────────────┬──────────────────────────────────────────┘
                   │ JAX-RS REST
┌──────────────────▼──────────────────────────────────────────┐
│  api/                                                        │
│  ├── MapResource.java     map data endpoints                │
│  ├── FetchResource.java   live API triggers + cooldown      │
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
                         │  └── ModelArtifacts  (loaders)   │
                         └──────────────────────────────────┘
                         ┌──────────────────────────────────┐
                         │  watcher/                         │
                         │  └── FileWatcherService           │
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
| `year_vocab.json` | Calendar year → embedding row index |
| `week_vocab.json` | ISO week number → embedding row index |
| `community_map.json` | H3 L8 hex → integer community ID |
| `h3_l9_neighbor_communities.json` | H3 hex → `int[7]` community indices (center + 6 neighbours) |
| `model_metadata.json` | Architecture dimensions, reference date, `n_communities` |

Key public methods:
- `scaleFeature(feature, value)` — `(value - mean) / scale`
- `inverseScaleLogPrice(scaled)` — recovers unscaled log price
- `getLogPriceScale()` — the log_price scaler std dev (used for uncertainty conversion)
- `lookupYear(year)` / `lookupWeek(week)` — vocab lookup with unknown fallback
- `lookupH3Neighbors(h3Index)` — returns `int[7]` neighbour community indices
- `lookupCommunity(h3Index)` — community ID string for display

#### `EmbeddingModel.java`
Wraps the ONNX session. Handles all feature engineering and inference.

The exported ONNX model has **6 inputs and 3 outputs**:

| Input | Shape | Description |
|---|---|---|
| `community_indices` | `[1, 7]` long | Center + 6 neighbour community indices |
| `year` | `[1]` long | Vocab-mapped year index |
| `week` | `[1]` long | Vocab-mapped ISO week index |
| `property_features` | `[1, 3]` float | Scaled sqft, sqft_lot, beds |
| `time_features` | `[1, 1]` float | Scaled time_trend |
| `market_features` | `[1, 2]` float | Scaled mortgage_rate, unemployment_rate |

| Output | Shape | Description |
|---|---|---|
| `log_price_scaled` | `[1, 1]` | Scaled log price |
| `log_var_scaled` | `[1, 1]` | Log variance from uncertainty head |
| `cls_attention` | `[1, 6]` | CLS token attention weights over 6 input tokens |

`predict()` returns a `PredictionResult` record:
```java
record PredictionResult(
    double  predictedPrice,      // $ price
    double  predictionStdPrice,  // $ std dev (delta method: price × std_log_price)
    double  predictionCvPct,     // 95% CI width as % of price (3.92 × std_log_price × 100)
    float[] clsAttention         // [community, year, week, property, time, market]
)
```

**Note on uncertainty calibration:** `predictionStdPrice` and `predictionCvPct` are only meaningful if the model was trained with `estimate_uncertainty=True` (NLL loss). With MSE-only training the uncertainty head exists but produces uncalibrated values. Retrain with NLL to get reliable confidence intervals.

---

### `ingest/` — data parsing and transformation

#### `PropertyRecord.java`
Immutable record — the single unified representation of a property across all sources.

Key fields:
- **Identity**: `id`, `address`, `source` (`"sales"`, `"rentcast"`, `"zillow"`)
- **Location**: `lat`, `lng`, `h3Index`, `community`
- **Property**: `sqft`, `sqftLot`, `beds`, `baths`, `homeType`
- **Transaction**: `saleDate`, `salePrice` (0 if unknown), `listingUrl`
- **Predictions**: `predictedPrice`, `pctError`
- **Uncertainty**: `predictionStdPrice`, `predictionCvPct`
- **Attention**: `clsAttention[6]` — CLS attention over [community, year, week, property, time, market]

#### `DataIngestionService.java`
Parses raw files and API responses into `PropertyRecord` lists. Calls `EmbeddingModel.predict()` per record at ingest time — all records in the store already have inference results.

| Method | Source |
|---|---|
| `ingestSalesCsv(file)` | `sales_2020_25.csv` — historical |
| `ingestRentcastCsv(file)` | RentCast CSV export |
| `ingestRentcastJson(file)` | RentCast JSON |
| `ingestZillowJson(file)` | Zillow JSON |
| `fetchFromRentcastApi()` | Live RentCast API |
| `fetchFromZillowApi()` | Live Zillow API |

#### `RentcastApiClient.java` / `ZillowApiClient.java`
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

`toGeoJson()` sets `displayValue` to whichever field matches the `variable` query param. Supported variables: `pct_error`, `sale_price`, `sqft`, `num_sales`, `pred_std`, `pred_cv_pct`, `attn_community`, `attn_year`, `attn_week`, `attn_property`, `attn_time`, `attn_market`.

---

### `api/` — REST endpoints

#### `MapResource.java`

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/rentcast` | GeoJSON points — sales (RentCast / historical / both, controlled by `source` param) | Sales points layer |
| `GET /api/zillow` | GeoJSON points — Zillow listings | Zillow layer |
| `GET /api/sales/h3` | GeoJSON polygons — H3 hex grid | H3 sales layer |
| `GET /api/sales/points` | GeoJSON points — raw sales for heatmap | Error heatmap |
| `GET /api/sales/community` | GeoJSON polygons — hexes by community | Community layer |
| `GET /api/performance` | JSON — quarterly error stats by community | Performance chart |
| `GET /api/home-types` | JSON — distinct home type strings | Property type dropdowns |
| `GET /api/stats` | JSON — summary counts and averages | Sidebar stats |

Both `/api/rentcast` and `/api/zillow` include `predictionStdPrice`, `predictionCvPct`, and all 6 `attn*` fields so the frontend can colour points by any variable.

The `/api/rentcast` endpoint accepts `source=rentcast|sales|all` to serve RentCast only, historical sales only, or both combined.

#### `FetchResource.java`
Triggers live API fetches. Both endpoints require `?confirm=true`. A configurable cooldown (`fetch.cooldown.hours`, default 6) prevents repeated calls. Coordinates with `FileWatcherService` via `markInProgress`/`markDone` to prevent double-ingest.

| Endpoint | Action |
|---|---|
| `POST /api/fetch/rentcast?confirm=true` | Calls RentCast, saves JSON, ingests |
| `POST /api/fetch/zillow?confirm=true` | Calls Zillow, saves JSON, ingests |
| `GET /api/fetch/status` | Last fetch times and cooldown state |

#### `ExportResource.java`
`GET /api/export/csv?source=all|sales|rentcast|zillow` — streams all in-memory records as a CSV download.

#### `StartupLoader.java`
Runs once at boot. Loads the configured historical sales CSV and any files already present in the watch directories.

---

### `watcher/` — filesystem monitoring

#### `FileWatcherService.java`
Watches `data/rentcast/` and `data/zillow/` using Java NIO `WatchService` on a background daemon thread. When a new file appears, waits for size stabilisation then ingests it. Skips files currently being written by `FetchResource`.

---

### `cli/` — standalone utilities

#### `RentcastFetcher.java`
Standalone `main()` for paginating through the RentCast API without the app running (`make rentcast-fetch`). Reads all parameters from `application.properties` and the API key from `.env`. Maintains a call counter in `.rentcast_call_count` (hard cap: 45 calls).

---

### `resources/` — static assets

#### `application.properties`

| Property | Default | Description |
|---|---|---|
| `data.sales.csv` | `../../data/sales_2020_25.csv` | Historical CSV loaded on startup |
| `watcher.rentcast.dir` | `data/rentcast` | Watched for new RentCast files |
| `watcher.zillow.dir` | `data/zillow` | Watched for new Zillow files |
| `fetch.cooldown.hours` | `6` | Minimum hours between live API calls (0 = disabled) |
| `rentcast.api.key` | *(from env)* | RentCast API key |
| `zillow.api.key` | *(from env)* | Zillow/HasData API key |

#### `model-artifacts/`
Exported from the Python pipeline by `export_model_for_java.py`:

| File | Description |
|---|---|
| `model.onnx` | PyTorch model with 3 outputs: price, log_var, cls_attention |
| `scalers.json` | Feature scaler parameters |
| `year_vocab.json` / `week_vocab.json` | Embedding index lookups |
| `community_map.json` | H3 hex → community ID |
| `h3_l9_neighbor_communities.json` | H3 hex → 7 community neighbour indices |
| `model_metadata.json` | Architecture dimensions, reference date, n_communities |

#### `META-INF/resources/index.html`
The entire frontend in a single file.

**Map layers:**

| Layer | Toggle | Colour driven by |
|---|---|---|
| H3 Sales | `showSales` | Variable selector (any variable) |
| Error Heatmap | `showHeat` | Replaces H3 layer, always shows error magnitude |
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

**Tooltips** show: address, sale/list price, predicted price, error %, 95% CI (`±$X ±Y%`), sqft/beds/baths, sale date, and all 6 CLS attention weights as percentages.

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

After retraining the Python model, run:
```bash
python export_model_for_java.py
```
This regenerates `model.onnx` (with all 3 outputs), `scalers.json`, `year_vocab.json`, `week_vocab.json`, `community_map.json`, and `model_metadata.json` in `java-app/model-artifacts/`. Copy them to `src/main/resources/model-artifacts/` and restart the app.

**For calibrated uncertainty**, train with `estimate_uncertainty=True` in `main_train_v4_h3l8.py`. This uses NLL loss which trains the `log_var` output head to produce meaningful variance estimates. Without this, `predictionStdPrice` and `predictionCvPct` will be present but unreliable.

---

## Adding a new map variable

1. Compute the per-record value during inference in `EmbeddingModel.predict()` or derive it in aggregation
2. Add it to `PropertyRecord` and `HexStats`
3. Compute the aggregate in `H3AggregationService.aggregateSales()`
4. Add a case to the `switch` in `toGeoJson()` and write it into `props`
5. Add `predictionCvPct` to both `/api/rentcast` and `/api/zillow` response properties
6. Add `<option>` to `<select id="variable">` in `index.html`
7. Add a case to `pointVariableValue()` and `getColorFn()` / legend `fmt`
