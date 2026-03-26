# house-price-app

Quarkus web app for Seattle house price prediction. Loads a trained ONNX model, ingests property data from files and live APIs, and serves an interactive Leaflet.js map.

---

## Running locally

```bash
cd java-app/house-price-app
mvn quarkus:dev
```

Open http://localhost:8080

---

## Project structure

```
house-price-app/
├── src/main/java/com/houseprices/
│   ├── model/
│   │   ├── ModelArtifacts.java       loads scalers, vocabs, H3 map from resources/
│   │   └── EmbeddingModel.java       ONNX inference + feature engineering
│   ├── ingest/
│   │   ├── PropertyRecord.java       unified data record (sales, rentcast, zillow)
│   │   ├── DataIngestionService.java reads CSVs/JSON and calls API clients
│   │   ├── RentcastApiClient.java    calls api.rentcast.io
│   │   └── ZillowApiClient.java      calls HasData Zillow scraper API
│   ├── service/
│   │   ├── PropertyStore.java        in-memory store, partitioned by source
│   │   └── H3AggregationService.java aggregates sales by H3 hex → GeoJSON polygons
│   ├── watcher/
│   │   └── FileWatcherService.java   watches drop directories for new files
│   └── api/
│       ├── MapResource.java          REST endpoints for the map frontend
│       ├── FetchResource.java        triggers live API fetches
│       └── StartupLoader.java        loads initial data files on boot
├── src/main/resources/
│   ├── application.properties
│   ├── model-artifacts/              ONNX model + JSON artifacts (bundled in JAR)
│   │   ├── model.onnx
│   │   ├── scalers.json
│   │   ├── community_vocab.json
│   │   ├── year_vocab.json
│   │   ├── week_vocab.json
│   │   ├── h3_l9_neighbor_communities.json
│   │   └── model_metadata.json
│   └── META-INF/resources/
│       └── index.html                Leaflet.js map frontend
└── pom.xml
```

---

## The data/ folder

`data/` here refers to **`java-app/house-price-app/data/`** — a runtime directory created when the app starts. It is not the project-root `data/` folder that holds the training CSVs.

It has two purposes:

**1. File drop directories for the watcher**

```
java-app/house-price-app/data/
├── rentcast/    ← drop new Rentcast CSV files here
└── zillow/      ← drop new Zillow JSON files here
```

`FileWatcherService` uses Java NIO `WatchService` to monitor these folders. When a new file appears it is ingested immediately and predictions are computed on the fly. No restart needed.

- Rentcast: expects CSV files in the same format as `rentcast_recent_house_sales.csv`
- Zillow: expects JSON files in the same format as `zillow_seattle_listings.json`

**2. Saved API fetch output**

When you click "Fetch Zillow API" or "Fetch RentCast API" in the UI, the raw response is written back to the startup data paths configured in `application.properties`:

```
../../data/zillow_seattle_listings.json      (project-root data/)
../../data/rentcast_recent_house_sales.csv
```

This means the next time the app starts, `StartupLoader` will automatically reload the most recently fetched data without needing another API call.

---

## Data flow

```
On startup
  StartupLoader
    → reads sales_2020_25.csv          (historical King County sales)
    → reads rentcast_recent_house_sales.csv
    → reads zillow_seattle_listings.json
    → runs ONNX inference on each record
    → stores all in PropertyStore (in memory)

While running
  FileWatcherService
    → watches data/rentcast/ and data/zillow/
    → new file dropped → ingest + predict → upsert into PropertyStore

  POST /api/fetch/rentcast or /api/fetch/zillow
    → calls live API
    → saves raw response to disk (persists across restarts)
    → ingests + predicts → upserts into PropertyStore

  GET /api/zillow          → Zillow listings as GeoJSON points
  GET /api/sales/h3        → sales aggregated by H3 hex as GeoJSON polygons
  GET /api/stats           → summary counts and averages
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/zillow` | Zillow listings as GeoJSON. Params: `maxError`, `minSqft`, `maxSqft` |
| GET | `/api/sales/h3` | H3-aggregated sales as GeoJSON polygons. Param: `variable` (`pct_error`, `sale_price`, `sqft`, `num_sales`) |
| GET | `/api/stats` | Summary statistics by source |
| POST | `/api/fetch/zillow` | Fetch live Zillow listings, save to disk, update map |
| POST | `/api/fetch/rentcast` | Fetch live RentCast sales, save to disk, update map |

---

## Configuration (application.properties)

| Key | Default | Description |
|-----|---------|-------------|
| `data.sales.csv` | `../../data/sales_2020_25.csv` | Historical sales CSV loaded on startup |
| `data.rentcast.csv` | `../../data/rentcast_recent_house_sales.csv` | RentCast CSV loaded on startup; also where API fetch saves |
| `data.zillow.json` | `../../data/zillow_seattle_listings.json` | Zillow JSON loaded on startup; also where API fetch saves |
| `watcher.rentcast.dir` | `data/rentcast` | Directory watched for new Rentcast CSV drops |
| `watcher.zillow.dir` | `data/zillow` | Directory watched for new Zillow JSON drops |
| `rentcast.api.key` | — | RentCast API key |
| `rentcast.lat/lng` | 47.60 / -122.33 | Search centre (Seattle) |
| `rentcast.radius` | 10 | Search radius in miles |
| `rentcast.months` | 6 | How far back to fetch sales |
| `zillow.api.key` | — | HasData Zillow scraper API key |
| `zillow.keyword` | `Seattle, WA` | Search location |
| `zillow.max.pages` | 10 | Max pages to fetch |
| `zillow.delay.ms` | 2000 | Delay between pages (rate limiting) |

---

## Model artifacts

Bundled inside the JAR under `model-artifacts/`. Generated by running `export_model_for_java.py` in the project root after training.

| File | Description |
|------|-------------|
| `model.onnx` | Exported PyTorch model (~950 KB) |
| `scalers.json` | StandardScaler mean/scale for each feature |
| `community_vocab.json` | Community ID → embedding index |
| `year_vocab.json` | Year → embedding index |
| `week_vocab.json` | ISO week → embedding index |
| `h3_l9_neighbor_communities.json` | H3 L9 hex → 7 community indices (center + 6 neighbours) |
| `model_metadata.json` | Architecture params, reference date, input/output spec |

To update the model after retraining:
```bash
# from project root
python3 export_model_for_java.py
cp -r java-app/model-artifacts/* java-app/house-price-app/src/main/resources/model-artifacts/
```

---

## Prediction pipeline

For each property the app computes:

1. H3 L9 index from lat/lng
2. Look up 7 community indices (center hex + 6 neighbours) from `h3_l9_neighbor_communities.json`
3. Scale continuous features using `scalers.json`
4. Map year and ISO week to vocab indices
5. Run ONNX inference → scaled log price
6. Inverse-scale → log price → `exp()` → predicted price in dollars
