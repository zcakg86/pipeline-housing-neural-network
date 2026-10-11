# OSM transport and water pipeline — pyosmium rewrite

## Design

This implementation chooses the **pyosmium + GeoPandas** route. It reads OSM PBF
files directly and retains real node IDs for topology. A PostgreSQL service,
osm2pgsql import, SQLAlchemy, and the GDAL `ogr2ogr` executable are not required.
SQLAlchemy would be useful for a separate PostGIS-backed architecture; it is not
needed to read local PBF files.

The implementation lives in the repository's `scripts/` directory:

- `extract_osm_transport.py`
- `extract_osm_water.py`
- `build_transport_features.py`
- `osm_common.py` (shared dependency)
- `osm_transport_config.ini`
- `osm_water_config.ini`

The two INI files contain Python extraction settings rather than GDAL OSM
driver settings.

## Install and run

Run from the repository root with Python 3.11+; validation used Python 3.12.
Create an environment if needed, then install the supplied requirements:

```bash
python -m pip install -r requirements.txt
python scripts/extract_osm_transport.py --input data/osm/washington-latest.osm.pbf
python scripts/extract_osm_water.py --input data/osm/washington-latest.osm.pbf
python scripts/build_transport_features.py
```

The original default input/output paths, deployment paths and feature names are
retained. Add `--no-deploy` to each command to write only canonical outputs.
The three scripts also work via `python -m scripts.<module_name>`.

Both extractors support:

```bash
python scripts/extract_osm_transport.py \
  --input data/osm/washington-latest.osm.pbf \
  --bbox -122.6 47.0 -121.0 47.9 \
  --simplify-degrees 0 \
  --no-deploy
```

`--bbox` specifies exact clipping bounds and overrides configuration. The
transport config carries the established buffered King County envelope as
`clip_bounds`, avoiding a separate location-indexed scan of the statewide PBF.
If neither is present, the extractor assembles the `King County`,
`admin_level=6` administrative relation, checks that exactly one matches, and
buffers its bounding rectangle by 0.05 degrees. Configurable boundary selection
is intended for extracts containing one matching county. It is a rectangle,
not a county-polygon mask.

`--config`, `--buffer-degrees`, `--simplify-degrees`, and `--location-index` are
available. The default location index is `flex_mem`. For a large extract, a
file-backed index can reduce RAM pressure:

```bash
python scripts/extract_osm_transport.py \
  --location-index sparse_file_array,/tmp/transport-node-locations.idx
```

Use separate cache paths for concurrent runs, and remove a file-backed cache
when finished. Nodes must remain in the source for geometry reconstruction;
do not prefilter the PBF down to tagged nodes only.

## Exact motorway connector definition

A point qualifies when both conditions hold **at the same OSM node ID**:

1. At least one incident edge belongs to a `highway=motorway_link` way.
2. A different incident edge belongs to an allowed non-motorway road way.

The shared node does not need a `highway=motorway_junction` tag. The output
records whether it has one in `taggedMotorwayJunction` for audit purposes.

Edges are pairs of consecutive way-node references. The junction can be an
interior vertex or an endpoint. A geometric crossing, identical coordinates
on different node IDs, or a nearby road does not establish a connection.
No ramp-chain traversal occurs.

The default allowed road classes are `trunk`, `primary`, `secondary`,
`tertiary`, their `_link` variants, `unclassified`, `residential`,
`living_street`, `service`, `road`, and `track`. Configure a subset with
`connector_road_types`. Motorways, other motorway links, paths, footways,
cycleways, construction and proposed roads cannot satisfy condition 3.

The connector's `osmId` now identifies the **junction node**, not the ramp way.
Added fields include `osmType`, `motorwayLinkWayIds`, `connectedRoadWayIds`,
`connectedRoadTypes`, and `incidentEdges` for audit. Original `kind`, `category`
and Point geometry remain. Actual shared nodes take precedence over differing
bridge/tunnel/layer tags: those tags can legitimately change at a connection.

**Interpretation:** the qualifying point is the actual node shared by a
motorway-link way and an allowed surface-road way. It may be tagged as a
motorway junction, but that is not required. Every transport output contains
counts for motorway-link nodes, qualifying connectors, tagged qualifying
connectors, and motorway-link nodes lacking an allowed road. An empty connector
layer also produces a terminal warning.

This is topological connectivity, not legal driving accessibility. Direction,
turn restrictions, private access, barriers and time-dependent restrictions do
not determine acceptance. Proximity remains straight-line distance.

## Transport, water and feature compatibility

- Transport route relations are read before member-way geometry extraction.
  Geometry consists of clipped member lines, never artificial lines joining
  disconnected pieces. Backward member roles are respected. Stop/platform way
  members are excluded from the route linework. Nested relation members are
  counted but not recursively expanded. Missing way references are counted and
  retained on route properties; routes with no local linework are omitted.
- Bus stops remain node-based. Rail stations can be nodes, ways or areas. The
  original preference for a named node over same-name/category outlines is
  retained. An outline uses a point on its geometry instead of its bounding-box
  centre. Distinct station nodes are retained, as before.
- Water retains the original `natural=water`, no `waterway` tag, and excluded
  subtype filters. Coastlines remain lines. pyosmium assembles closed ways and
  multipolygon relations, preserving holes. Shapely repairs invalid geometries
  before clipping and removes lower-dimensional clipping debris. Area assembly
  still requires complete relation/way/node membership in the source extract.
- OSM IDs remain string fields; `osmType` disambiguates node/way/relation IDs.
- H3 distances and road lengths now use **EPSG:32610 (WGS84 / UTM zone 10N)** via
  GeoPandas/PyProj, replacing the fixed-latitude approximation. This CRS is for
  the King County workflow; choose another appropriate metric CRS for other regions.
- The seven H3 fields and output structure remain, but `schema_version` is **2**
  because connector semantics and distance calculations changed. Check any
  downstream code that enforces schema version 1. The fitted bus p95 normalizer
  and metric CRS are recorded in metadata. Rebuild all three artifacts together.
- Proximity arrays are computed once, removing repeated whole-array work per
  cell. Duplicate grid IDs and invalid coordinates raise errors. Empty source
  layers retain the original zero-score behavior; inspect source counts to
  distinguish missing supply from missing data.

The existing major-road density still counts OSM carriageway length, not a
collapsed road-centreline network. Nearby bus route count still uses `route_ref`
on stop nodes, not inferred route membership. Simplification remains in degrees
for output compatibility; use `--simplify-degrees 0` for unsimplified analytical
layers. Ensure clip bounds cover grid cells plus at least the 1 km density
catchment; nearest-feature scores can also be biased near extraction edges.

## Resource use and audit

The implementation makes sequential passes: an optional area-enabled boundary
scan, a transport catalogue pass, and an area-enabled geometry scan. pyosmium's
area handling itself uses two file scans. It does not materialize the statewide
road graph as a GeoDataFrame. Python memory holds tagged candidate junctions,
route membership, selected local route geometries, and final output features;
the node-location index can still be the largest memory consumer.

Invalid geometry callbacks and missing route members are counted. Incomplete
PBF node locations may cause pyosmium's location handler to fail before a
callback, intentionally stopping extraction. Area-assembly failures may also
be logged by libosmium rather than included in custom callback counters: those
counters are not a proof of source completeness.

## Verification

```bash
python -m unittest tests.test_osm_pipeline -v
```

Six integration tests generate an OSM fixture, convert it to binary PBF, and
exercise the real readers. They cover strict node identity; an interior-way
junction; exclusion of untagged nodes, footways, motorway-only connections and
coordinate-only crossings; a valid shared node at a bridge-tag transition;
route members and missing references; water polygons and multipolygon holes;
county boundary assembly; metric H3 features and empty layers; and both
extractor CLIs with default boundary lookup.

All six passed using the pinned requirements. No Washington production PBF or
actual H3 grid was supplied, so production counts, runtime, peak memory and
Java integration have not been verified.

References:
- https://docs.osmcode.org/pyosmium/latest/reference/Handler-Processing/
- https://wiki.openstreetmap.org/wiki/Tag:highway=motorway_junction
- https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
