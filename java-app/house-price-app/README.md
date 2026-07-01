# House Price App — Component Guide

A Quarkus application that loads property data, runs ONNX inference, and serves an interactive Leaflet.js map. This guide explains every source file, what it does, and how the components connect.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (index.html + Leaflet.js)                          │
│  ├── GET /api/sales/h3       → H3 hex layer (polygons)      │
│  ├── GET /api/sales/points   → heatmap layer (points)       │
│  ├── GET /api/rentcast       → RentCast sales points        │
│  ├── GET /api/zillow         → Zillow listing points        │
│  ├── GET /api/performance    → community error chart        │
│  ├── GET /api/stats          → sidebar stats                │
│  ├── GET /api/export/csv     → CSV download                 │
│  └── POST /api/fetch/rentcast|zillow → live API fetch       │
└──────────────────┬──────────────────────────────────────────┘
                   │ JAX-RS REST
┌──────────────────▼──────────────────────────────────────────┐
│  api/  — REST layer                                          │
│  ├── MapResource.java    (map data endpoints)               │
│  ├── FetchResource.java  (live API triggers)                │
│  ├── ExportResource.java (CSV download)                     │
│  └── StartupLoader.java  (data load on boot)                │
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
                         │  ├── EmbeddingModel    (ONNX)    │
                         │  └── ModelArtifacts   (loaders)  │
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
Loads all static model files from `src/main/resources/model-artifacts/` at startup (`@PostConstruct`). Holds them in memory for the lifetime of the application.

| File loaded | What it contains |
|---|---|
| `scalers.json` | Mean and std dev for every scaled feature. Used to normalise inputs before inference and to inverse-transform the log-price output. |
| `year_vocab.json` | Calendar year → embedding table row index (e.g. `"2024" → 4`). |
| `week_vocab.json` | ISO week number → embedding table row index. |
| `community_map.json` | H3 L8/L9 hex string → integer community ID. Used to attach a community label to each property. |
| `h3_l9_neighbor_communities.json` | H3 L9 hex → 7-element list of community indices (center + 6 neighbours). Fed directly into the model's community embedding layer. |
| `model_metadata.json` | Architecture dimensions, reference date, `n_communities`. The reference date anchors the `time_trend` feature; `n_communities` gives the unknown community index. |

Key public methods:
- `scaleFeature(feature, value)` — applies `(value - mean) / scale`
- `inverseScaleLogPrice(scaled)` — recovers unscaled log price
- `getLogPriceScale()` — exposes the log_price std dev (needed for uncertainty conversion)
- `lookupYear(year)` / `lookupWeek(week)` — vocab lookups with fallback to unknown index
- `lookupH3Neighbors(h3Index)` — returns the `int[7]` neighbor community indices
- `lookupCommunity(h3Index)` — returns the community ID string for display

#### `EmbeddingModel.java`
Wraps the ONNX session. Handles all feature engineering and runs inference.

The ONNX model has 6 inputs and 3 outputs:

| Input | Shape | Description |
|---|---|---|
| `community_indices` | `[1, 7]` long | Center + 6 neighbour community indices |
| `year` | `[1]` long | Vocab-mapped year index |
| `week` | `[1]` long | Vocab-mapped ISO week index |
| `property_features` | `[1, 3]` float | Scaled sqft, sqft_lot, beds |
| `time_features` | `[1, 1]` float | Scaled time_trend (days since reference / 365.25) |
| `market_features` | `[1, 2]` float | Scaled mortgage_rate, unemployment_rate |

| Output | Shape | Description |
|---|---|---|
| `log_price_scaled` | `[1, 1]` | Scaled log price — inverse-scaled then exp'd to get $ price |
| `log_var_scaled` | `[1, 1]` | Log variance from uncertainty head — converted to price std dev |
| `cls_attention` | `[1, 6]` | CLS token attention weights over the 6 input tokens |

`predict()` returns a `PredictionResult` record:
```java
record PredictionResult(
    double  predictedPrice,      // $ price
    double  predictionStdPrice,  // $ std dev (≈ price × std_log_price, delta method)
    float[] clsAttention         // [community, year, week, property, time, market]
)
```

The method gracefully falls back to zeros for `predictionStdPrice` and `clsAttention` if the loaded ONNX model only has 1 output (older export).

---

### `ingest/` — data parsing and transformation

#### `PropertyRecord.java`
A Java record (immutable value type) that is the single unified representation of a property across all data sources. Every ingest method produces `PropertyRecord` instances; every API endpoint reads them from the store.

Key fields:
- **Identity**: `id`, `address`, `source` (`"sales"`, `"rentcast"`, `"zillow"`)
- **Location**: `lat`, `lng`, `h3Index`, `community`
- **Property**: `sqft`, `sqftLot`, `beds`, `baths`, `homeType`
- **Transaction**: `saleDate`, `salePrice` (0 if unknown), `listingUrl`
- **Predictions**: `predictedPrice`, `pctError` (% error vs sale price)
- **Uncertainty/attention**: `predictionStdPrice`, `clsAttention[6]`

#### `DataIngestionService.java`
Parses raw files and API responses into `PropertyRecord` lists. Has one method per source format:

| Method | Input format | Source tag |
|---|---|---|
| `ingestSalesCsv(file)` | `sales_2020_25.csv` — historical sales | `"sales"` |
| `ingestRentcastCsv(file)` | RentCast CSV export | `"rentcast"` |
| `ingestRentcastJson(file)` | RentCast JSON (from API or CLI fetch) | `"rentcast"` |
| `ingestZillowJson(file)` | Zillow JSON (from API fetch or file drop) | `"zillow"` |
| `fetchFromRentcastApi()` | Live RentCast API call | `"rentcast"` |
| `fetchFromZillowApi()` | Live Zillow API call | `"zillow"` |

Every method calls `EmbeddingModel.predict()` per record to produce `predictedPrice`, `predictionStdPrice`, `pctError`, and `clsAttention` at ingest time. There is no lazy prediction — all records in the store already have inference results.

#### `RentcastApiClient.java`
Makes HTTP calls to `api.rentcast.io/v1/properties`. Parameters (lat, lng, radius, months, limit) come from `application.properties`. Throws `IllegalStateException` if the API key is not configured. Returns raw `List<Map<String, Object>>` which `DataIngestionService` then parses into `PropertyRecord`s.

#### `ZillowApiClient.java`
Same pattern as `RentcastApiClient` but calls the Zillow/HasData API. Supports pagination with a configurable `maxPages` cap and a configurable delay between pages to avoid rate limiting.

---

### `service/` — in-memory storage and aggregation

#### `PropertyStore.java`
The application's in-memory database. Stores all `PropertyRecord`s in a `ConcurrentHashMap<source, ConcurrentHashMap<id, record>>`. Thread-safe by design — records can be upserted from the file watcher thread while the API serves reads.

Key methods:
- `upsert(records)` — insert or overwrite by `(source, id)` — deduplicates on re-ingest
- `getBySource(source)` — all records for one source
- `getSalesRecords()` — `"sales"` + `"rentcast"` combined (both have actual sale prices)
- `getZillowListings()` — `"zillow"` only
- `countsBySource()` — map of source → count, used in the stats endpoint

#### `H3AggregationService.java`
Groups filtered `PropertyRecord`s by their `h3Index` and computes hex-level statistics for the map layer.

`HexStats` record fields per hex:

| Field | Description |
|---|---|
| `avgSalePrice` | Mean actual sale price |
| `avgPredictedPrice` | Mean predicted price |
| `avgPctError` | Mean % prediction error |
| `avgSqft` | Mean living area |
| `numSales` | Record count |
| `avgPredStd` | Mean prediction uncertainty in $ |
| `attnCommunity..attnMarket` | Mean CLS attention per token |
| `boundary` | GeoJSON polygon ring for the hex |

`toGeoJson()` converts `HexStats` to a GeoJSON `FeatureCollection`. The `displayValue` field in each feature's properties is set to whichever field matches the `variable` query parameter — this is what the frontend uses to colour the hexes.

Supported `variable` values: `pct_error`, `sale_price`, `sqft`, `num_sales`, `pred_std`, `attn_community`, `attn_year`, `attn_week`, `attn_property`, `attn_time`, `attn_market`.

---

### `api/` — REST endpoints

#### `MapResource.java`
The main map API. All endpoints return GeoJSON.

| Endpoint | Returns | Used by |
|---|---|---|
| `GET /api/rentcast` | Point features — RentCast sales | Rentcast layer (green dots) |
| `GET /api/zillow` | Point features — Zillow listings | Zillow layer (blue dots) |
| `GET /api/sales/h3` | Polygon features — H3 hex grid | H3 sales layer |
| `GET /api/sales/points` | Point features — raw sales for heatmap | Error heatmap |
| `GET /api/sales/community` | Polygon features — hexes coloured by community | Community layer |
| `GET /api/performance` | JSON — quarterly error stats by community | Performance chart popup |
| `GET /api/home-types` | JSON — distinct home type strings | Property type filter dropdowns |
| `GET /api/stats` | JSON — summary counts and averages | Sidebar stats panel |

Most endpoints accept filter query parameters: `minError`, `maxError`, `minSqft`, `maxSqft`, `homeType`, `dateFrom`, `dateTo`.

#### `FetchResource.java`
Triggers live API fetches. Both endpoints require `?confirm=true` to prevent accidental billed calls. A configurable cooldown (`fetch.cooldown.hours`, default 6) blocks repeated calls within a short window.

| Endpoint | Action |
|---|---|
| `POST /api/fetch/rentcast?confirm=true` | Calls RentCast API, saves JSON to `data/rentcast/`, ingests into store |
| `POST /api/fetch/zillow?confirm=true` | Calls Zillow API, saves JSON to `data/zillow/`, ingests into store |
| `GET /api/fetch/status` | Shows last fetch times and cooldown state |

The fetch methods coordinate with `FileWatcherService` via `markInProgress`/`markDone` to prevent the watcher from double-ingesting files the fetch just wrote.

#### `ExportResource.java`
`GET /api/export/csv?source=all|sales|rentcast|zillow` — streams all in-memory records as a CSV download, one row per `PropertyRecord`.

#### `StartupLoader.java`
Runs once at application start (`@Observes StartupEvent`). Loads the configured historical sales CSV (`data.sales.csv`) and any files already present in the rentcast and zillow watch directories. This is what populates the store before any live API fetch has happened.

---

### `watcher/` — filesystem monitoring

#### `FileWatcherService.java`
Watches `data/rentcast/` and `data/zillow/` for new files using the Java NIO `WatchService`. Runs on a background daemon thread. When a new file appears, it waits for the file to finish writing (size stabilisation), then ingests it and upserts the records into `PropertyStore`.

Coordinates with `FetchResource` via an `inProgress` set to avoid processing files that the fetch endpoints are currently writing — otherwise a file would be ingested twice (once by the watcher, once by the fetch endpoint itself).

Supported file types:
- `data/rentcast/*.json` → `ingestRentcastJson`
- `data/rentcast/*.csv` → `ingestRentcastCsv`
- `data/zillow/*.json` → `ingestZillowJson`

---

### `cli/` — standalone utilities

#### `RentcastFetcher.java`
A `main()` class that can be run outside the Quarkus app (via `make rentcast-fetch`) to paginate through the RentCast API and save JSON files to `data/rentcast/`. Reads all parameters from `application.properties` (same file as the app) and the API key from `.env`. Maintains a call counter in `.rentcast_call_count` with a hard cap (`MAX_CALLS = 45`) to prevent runaway billing.

---

### `resources/` — static assets

#### `application.properties`
Central configuration. Key properties:

| Property | Default | Description |
|---|---|---|
| `data.sales.csv` | `../../data/sales_2020_25.csv` | Historical sales CSV loaded on startup |
| `watcher.rentcast.dir` | `data/rentcast` | Directory watched for new RentCast files |
| `watcher.zillow.dir` | `data/zillow` | Directory watched for new Zillow files |
| `fetch.cooldown.hours` | `6` | Minimum hours between live API calls (0 = disabled) |
| `rentcast.api.key` | *(from env)* | RentCast API key |
| `rentcast.lat/lng/radius/months/limit` | Seattle defaults | RentCast search parameters |
| `zillow.api.key` | *(from env)* | Zillow/HasData API key |

#### `model-artifacts/`
Static files copied from the Python pipeline by `export_model_for_java.py`:

| File | Description |
|---|---|
| `model.onnx` | Exported PyTorch model with 3 outputs |
| `scalers.json` | Feature scaler parameters |
| `year_vocab.json` / `week_vocab.json` | Embedding index lookups |
| `community_map.json` | H3 hex → community ID |
| `h3_l9_neighbor_communities.json` | H3 hex → 7 community neighbor indices |
| `model_metadata.json` | Architecture dimensions, reference date, n_communities |

#### `META-INF/resources/index.html`
The entire frontend in a single file. Uses Leaflet.js for the map and Chart.js for the performance chart. Fetches all data from the REST API and renders it client-side. Key sections:

- **Layer toggles** — checkboxes to show/hide the Zillow, RentCast, H3 sales, heatmap, and community layers
- **Variable dropdown** — selects which field to colour the H3 hexes by (error, price, sqft, sales count, uncertainty, or any of the 6 attention weights)
- **Error filter** — sliders with optional "No filter" checkboxes to filter records by % error
- **Sqft filter** — sliders for property size
- **Date range** — filters historical sales by sale date
- **`errorColor(v, anchors)`** — diverging blue/grey/red colour function anchored to the p10 and p90 of the loaded data
- **`loadSales()`** — the main map refresh function; fetches H3 or heatmap data and redraws
- **`loadCommunityLayer()`** — loads community polygons and the performance chart data
- **Performance popup** — draggable overlay with a per-community quarterly error chart

---

## Data flow on a request

```
Browser clicks "Fetch RentCast API"
  → POST /api/fetch/rentcast?confirm=true
      FetchResource.fetchRentcast()
        RentcastApiClient.fetchRecentSales()     ← HTTP to api.rentcast.io
        saves JSON to data/rentcast/rentcast_latest.json
        FileWatcherService.markInProgress(path)  ← prevents double-ingest
        DataIngestionService.ingestRentcastJson(file)
          for each property:
            EmbeddingModel.predict(h3, date, sqft, sqftLot, beds)
              ModelArtifacts.lookupH3Neighbors()
              ModelArtifacts.scaleFeature() × 6
              OrtSession.run() → [log_price, log_var, cls_attention]
              → PredictionResult(price, stdPrice, attn[6])
          → List<PropertyRecord>
        PropertyStore.upsert(records)
        FileWatcherService.markDone(path)
  ← { status: "ok", fetched: 342 }

Browser refreshes map
  → GET /api/sales/h3?variable=pct_error&...
      MapResource.salesH3()
        H3AggregationService.aggregateSales()
          PropertyStore.getSalesRecords()       ← in-memory, O(1)
          group by h3Index
          compute avgPctError, avgPredStd, avgAttn[6], boundary polygon
          → List<HexStats>
        H3AggregationService.toGeoJson()
          set displayValue = avgPctError
          write all stats into feature properties
  ← GeoJSON FeatureCollection of H3 polygons
```

---

## Adding a new data source

1. Add a parse method in `DataIngestionService` that returns `List<PropertyRecord>`
2. Call `EmbeddingModel.predict()` per record and populate all prediction fields
3. Load it in `StartupLoader` or add a new watch directory in `FileWatcherService`
4. Add a `getBySource("new_source")` accessor to `PropertyStore` if needed
5. Include it in `getSalesRecords()` if it has actual sale prices (so it appears in the H3 layer and performance chart)

## Adding a new map variable

1. Compute the per-record field in `EmbeddingModel.predict()` or derive it in aggregation
2. Add it to `PropertyRecord` (if per-record) and `HexStats` (aggregated)
3. Compute the aggregate in `H3AggregationService.aggregateSales()`
4. Add a case to the `switch` in `toGeoJson()` and write it into `props`
5. Add an `<option>` to `<select id="variable">` in `index.html`
6. Update `getColorFn()` and `buildLegend()` if it needs a different colour scale
