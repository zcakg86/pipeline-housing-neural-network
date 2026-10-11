"""Shared pyosmium readers and GeoJSON/config helpers (no GDAL CLI required)."""
from __future__ import annotations
import argparse
import configparser
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import osmium
from shapely import make_valid, from_wkb
from shapely.geometry import box, mapping, MultiLineString, MultiPolygon

FACTORY = osmium.geom.WKBFactory()
SURFACE_ROADS = {
    'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary',
    'secondary_link', 'tertiary', 'tertiary_link', 'unclassified',
    'residential', 'living_street', 'service', 'road', 'track',
}


def csv_set(value):
    return {s.strip() for s in value.split(',') if s.strip()}


def load_config(path):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path):
        raise FileNotFoundError(path)
    if not config.has_section('extract'):
        raise ValueError(f'{path}: expected new [extract] config, not GDAL OSM config')
    return config


def add_extract_arguments(parser, default_config):
    parser.add_argument('--config', type=Path, default=default_config)
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('WEST','SOUTH','EAST','NORTH'),
                        help='Explicit clip bounds; bypass boundary lookup and buffer.')
    parser.add_argument('--buffer-degrees', type=float, default=None)
    parser.add_argument('--simplify-degrees', type=float, default=None)
    parser.add_argument('--location-index', default='flex_mem',
                        help='pyosmium node location index, e.g. sparse_file_array,/tmp/nodes.idx')
    parser.add_argument('--no-deploy', action='store_true')


def geometry(obj, area=False):
    raw = FACTORY.create_multipolygon(obj) if area else FACTORY.create_linestring(obj)
    return from_wkb(raw)


def typed_geometry(value, dimension):
    """Discard lower-dimensional debris after repair and clipping."""
    if value.is_empty:
        return None
    accepted = {'line': ('LineString', 'MultiLineString'),
                'polygon': ('Polygon', 'MultiPolygon')}[dimension]
    if value.geom_type in accepted:
        return value
    parts = []
    for part in getattr(value, 'geoms', []):
        selected = typed_geometry(part, dimension)
        if selected is not None:
            parts.extend(selected.geoms if selected.geom_type.startswith('Multi') else [selected])
    if not parts:
        return None
    return MultiLineString(parts) if dimension == 'line' else MultiPolygon(parts)


def clipped(value, bounds, tolerance, dimension):
    if not value.is_valid:
        value = make_valid(value)
    if not value.intersects(box(*bounds)):
        return None
    value = typed_geometry(value.intersection(box(*bounds)), dimension)
    if value is not None and tolerance:
        value = typed_geometry(value.simplify(tolerance, preserve_topology=True), dimension)
    return value


class Boundary(osmium.SimpleHandler):
    def __init__(self, name, admin_level):
        super().__init__()
        self.name, self.admin_level = name, admin_level
        self.matches = {}

    def area(self, area):
        if (not area.from_way() and area.tags.get('boundary') == 'administrative'
                and area.tags.get('name') == self.name
                and area.tags.get('admin_level') == self.admin_level):
            self.matches[area.orig_id()] = geometry(area, area=True).bounds


def extraction_options(args, config):
    section = config['extract']
    buffer = args.buffer_degrees if args.buffer_degrees is not None else section.getfloat('buffer_degrees')
    tolerance = args.simplify_degrees if args.simplify_degrees is not None else section.getfloat('simplify_degrees')
    if buffer < 0 or tolerance < 0:
        raise ValueError('Buffer and simplification must be nonnegative')
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    original = None
    configured_bounds = section.get('clip_bounds', '').strip()
    if args.bbox:
        bounds = tuple(args.bbox)
    elif configured_bounds:
        values = [float(value.strip()) for value in configured_bounds.split(',')]
        if len(values) != 4:
            raise ValueError('clip_bounds must contain west,south,east,north')
        bounds = tuple(values)
    else:
        boundary = Boundary(section.get('boundary_name', 'King County'), section.get('admin_level', '6'))
        boundary.apply_file(str(args.input), locations=True, idx=args.location_index)
        if len(boundary.matches) != 1:
            raise ValueError(f'Expected one boundary, found {len(boundary.matches)}; use --bbox')
        original = next(iter(boundary.matches.values()))
        w,s,e,n = original
        bounds = w-buffer,s-buffer,e+buffer,n+buffer
    w,s,e,n = bounds
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f'Invalid clip bounds: {bounds}')
    return bounds, tolerance, original, buffer


def raw_feature(value, tags, osm_id, osm_type):
    return {'type':'Feature','geometry':mapping(value),
            'properties':{**tags,'osm_id':str(osm_id),'osm_type':osm_type}}


def write_collection(path, features, *, source, bounds, simplify_degrees, audit=None, metadata=None):
    payload = {'type':'FeatureCollection', 'metadata':{
        'source':source.name, 'generatedAt':datetime.now(timezone.utc).isoformat(),
        'extractor':'pyosmium', 'clipBounds':list(bounds), 'simplifyDegrees':simplify_degrees,
        'featureCount':len(features),
        'countsByKind':dict(Counter(f['properties'].get('kind','water') for f in features)),
        'countsByCategory':dict(Counter(f['properties']['category'] for f in features)),
        'audit':audit or {}, **(metadata or {})}, 'features':features}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(f'Wrote {len(features):,} features: {path}')


def deploy(source, target):
    if source.resolve() != target.resolve():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
