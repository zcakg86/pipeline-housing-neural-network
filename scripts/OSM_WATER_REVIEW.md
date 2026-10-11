# Review of the original water extraction logic

The original `extract_osm_water.py` and `osm_water_config.ini` are a reasonable
compact map-overlay pipeline. They are not a complete hydrological layer or,
by themselves, a water-accessibility feature builder. No water proximity,
coverage, view, frontage, flood or access calculation is present in these files.

## What is sound

The script extracts polygons and coastlines separately, uses a buffered county
bounding box to reduce edge truncation, preserves useful water subtypes and
intermittency tags, repairs geometry before final clipping, records output
counts and parameters, and writes a small deployment artifact. These are sensible
choices for an exploratory map.

## Material issues

1. **The sea is not a polygon in the result.** `natural=coastline` ways produce
   linework. This supports drawing a coastline and measuring distance to that
   line, but not marine-water area fractions, land/water containment, or complete
   water masks. For those, add processed marine-water polygons; do not close
   clipped coastline segments arbitrarily.
2. **The inland-water filter is driven partly by tagging style.** The SQL
   requires `natural='water' AND waterway IS NULL`. Therefore every polygon with
   a `waterway` value is rejected, regardless of its shape or useful water type.
   Modern river polygons tagged `natural=water, water=river` can pass; it is
   incorrect to say that all rivers are excluded. Legacy `waterway=riverbank`
   polygons do not pass. The INI tells GDAL how to recognize these as polygons,
   but the subsequent SQL still filters them out. Supporting that legacy tag
   requires care: some remaining uses represent intermittent riverbeds.
3. **The apparent water-type blacklist is narrow.** Excluding water values
   `stream`, `drain`, `ditch`, and `weir` does not establish a general policy for
   drainage features; these concepts commonly appear under `waterway`. There
   is no rule based on size, permanence, or amenity. Small ponds and artificial
   water bodies can remain. Waterway centreline rivers/streams and wetlands
   are not extracted.
4. **`natural_water` overstates the classification.** `natural=water` is an OSM
   mapping category, not proof of natural origin. The script's selection can
   include reservoirs, artificial ponds and other water bodies. A neutral
   category such as `inland_water`, plus a retained subtype, is more accurate.
   Changing the existing category should be coordinated with the map consumer.
5. **Display processing is not an analytical reference geometry.** The original
   applies degree-based simplification before its later make-valid stage. The
   tolerance is small (roughly 2–3 metres locally), but not a uniform metric
   tolerance. Geometry can be lost before repair, with `-skipfailures` allowing
   extraction to continue. Later counts cannot reveal how much was omitted.
   The rewrite repairs before clipping/simplification and counts callback
   failures, but pyosmium assembly failures still require log review.
6. **Geometry provenance is limited.** The original stores a bare OSM ID without
   its element type, and the input filename plus processing time rather than
   the source snapshot timestamp/hash. Node, way and relation ID namespaces are
   distinct. The rewrite adds `osmType`; it does not establish historical
   source validity. If used in historical price modelling, record the source
   snapshot and decide explicitly whether present-day geography is acceptable.

## Recommended policy for a house-price model

Keep the lightweight map output, but derive analytical features from a separate
unsimplified or metre-tolerance layer. Separate inland-water polygons,
marine-water polygons/coastline, and river/stream centreline features. Preserve
subtype, intermittent/seasonal status, original tags and typed OSM identity.
Choose an explicit inclusion policy based on the intended signal.

For example, distinguish distance to lake/reservoir, coast, and river from the
water-covered fraction inside a local buffer. Retain water-body area so a small
pond does not automatically stand in for a major lake. Nearest-water distance
alone does not establish waterfront frontage, views, public access, or flood
risk; those need additional evidence and definitions.

If deriving a shoreline from polygons, use the original water geometry before
clipping, or mark/remove artificial clip-boundary edges. Otherwise the clipped
rectangle can create a false shoreline. Dissolve adjoining polygons belonging
to the same body before calculating shoreline distance to avoid internal seams.

The current rewrite intentionally preserves the original selection filter and
category names. It improves the extraction engine and geometry handling without
silently choosing a new water policy. Real-world omission counts cannot be
estimated without the actual PBF and a comparison layer.

## Source documentation

- OSM river-area tags and legacy riverbank handling:
  https://wiki.openstreetmap.org/wiki/Tag:waterway=riverbank
  https://wiki.openstreetmap.org/wiki/Tag:water=river
- Coastline semantics:
  https://wiki.openstreetmap.org/wiki/Tag:natural=coastline
- Processed marine-water polygons:
  https://osmdata.openstreetmap.de/data/water-polygons.html
