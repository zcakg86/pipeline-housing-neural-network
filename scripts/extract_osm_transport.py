#!/usr/bin/env python3
"""Extract compact King County public-transport and major-road map layers with pyosmium."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from shapely.geometry import shape, mapping, MultiLineString, Point

import osmium

if __package__:
    from .osm_common import (add_extract_arguments, load_config, extraction_options, csv_set, geometry, clipped, raw_feature, write_collection, deploy, SURFACE_ROADS)
else:
    from osm_common import (add_extract_arguments, load_config, extraction_options, csv_set, geometry, clipped, raw_feature, write_collection, deploy, SURFACE_ROADS)

DEFAULT_INPUT = Path("data/osm/washington-latest.osm.pbf")
DEFAULT_OUTPUT_DIR = Path("data/osm")
DEFAULT_DEPLOY_DIR = Path(
    "java-app/house-price-app/src/main/resources/META-INF/resources/data"
)

RAIL_STATION_TYPES = {"station", "halt", "tram_stop"}
RAIL_ROUTE_TYPES = {"light_rail", "tram", "monorail", "train"}
BUS_ROUTE_TYPES = {"bus", "trolleybus"}
MAJOR_ROAD_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
}


def text(properties: dict[str, Any], name: str) -> str:
    value = properties.get(name)
    return "" if value is None else str(value)


def yes(properties: dict[str, Any], name: str) -> bool:
    return text(properties, name).lower() in {"yes", "true", "1"}


def route_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def classify_rail_station(properties: dict[str, Any]) -> str:
    station = text(properties, "station").lower()
    railway = text(properties, "railway").lower()
    if yes(properties, "monorail") or station == "monorail":
        return "monorail"
    if yes(properties, "light_rail") or station == "light_rail":
        return "light_rail"
    if yes(properties, "tram") or station == "tram" or railway == "tram_stop":
        return "tram"
    if yes(properties, "subway") or station == "subway":
        return "subway"
    if yes(properties, "train") or railway in {"station", "halt"}:
        return "train"
    return "rail_station"


def coordinate_pairs(value: Any):
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, list):
        for child in value:
            yield from coordinate_pairs(child)


def representative_point(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return points unchanged; use a point on the geometry for outlines."""
    if not geometry:
        return geometry
    if geometry.get("type") == "Point":
        return geometry
    return mapping(shape(geometry).representative_point())


def normalize_station(feature: dict[str, Any], mode: str) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    if mode == "rail":
        category = classify_rail_station(properties)
    elif text(properties, "amenity") == "bus_station" or \
            text(properties, "public_transport") == "station":
        category = "bus_station"
    elif yes(properties, "trolleybus"):
        category = "trolleybus_stop"
    else:
        category = "bus_stop"
    return {
        "type": "Feature",
        "geometry": representative_point(feature.get("geometry")),
        "properties": {
            "kind": "station",
            "category": category,
            "osmId": text(properties, "osm_id") or text(properties, "osm_way_id"),
            "osmType": text(properties, "osm_type"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref") or text(properties, "local_ref"),
            "railway": text(properties, "railway"),
            "publicTransport": text(properties, "public_transport"),
            "routes": route_list(text(properties, "route_ref")),
            "network": text(properties, "network"),
            "operator": text(properties, "operator"),
            "wheelchair": text(properties, "wheelchair"),
            "shelter": text(properties, "shelter"),
            "bench": text(properties, "bench"),
        },
    }


def normalize_route(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    route_type = text(properties, "route").lower()
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            "kind": "route",
            "category": f"{route_type}_route",
            "routeType": route_type,
            "osmId": text(properties, "osm_id"),
            "osmType": text(properties, "osm_type"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref"),
            "from": text(properties, "from"),
            "to": text(properties, "to"),
            "via": text(properties, "via"),
            "network": text(properties, "network"),
            "operator": text(properties, "operator"),
            "colour": text(properties, "colour"),
        },
    }


def normalize_road(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    road_type = text(properties, "highway").lower()
    category = road_type.removesuffix("_link")
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            "kind": "road",
            "category": category,
            "roadType": road_type,
            "link": road_type.endswith("_link"),
            "osmId": text(properties, "osm_id"),
            "osmType": text(properties, "osm_type"),
            "name": text(properties, "name"),
            "ref": text(properties, "ref"),
            "lanes": text(properties, "lanes"),
            "maxspeed": text(properties, "maxspeed"),
            "surface": text(properties, "surface"),
            "oneway": text(properties, "oneway"),
            "bridge": text(properties, "bridge"),
            "tunnel": text(properties, "tunnel"),
            "access": text(properties, "access"),
            "busway": text(properties, "busway"),
        },
    }



def is_rail(tags):
    return tags.get('railway') in RAIL_STATION_TYPES


def is_bus(tags):
    return (tags.get('highway') == 'bus_stop' or tags.get('amenity') == 'bus_station'
            or (tags.get('public_transport') in {'platform','station'}
                and (yes(tags,'bus') or yes(tags,'trolleybus'))))


class Catalogue(osmium.SimpleHandler):
    """First pass: only candidate nodes and route membership, no way geometry."""
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.junctions = {}
        self.motorway_link_node_ids = set()
        self.stations = []
        self.routes = {}
        self.route_way_ids = set()
        self.audit = Counter()

    def node(self, node):
        tags = dict(node.tags)
        if not (tags.get('highway') == 'motorway_junction' or is_rail(tags) or is_bus(tags)):
            return
        if not node.location.valid():
            self.audit['invalid_candidate_node_locations'] += 1
            return
        x,y = node.location.lon,node.location.lat
        w,s,e,n = self.bounds
        if not (w <= x <= e and s <= y <= n):
            return
        from shapely.geometry import Point
        feature = raw_feature(Point(x,y),tags,node.id,'node')
        if tags.get('highway') == 'motorway_junction':
            self.junctions[node.id] = feature
        if is_rail(tags) or is_bus(tags):
            self.stations.append(feature)

    def way(self, way):
        """Remember every motorway-link node for topology matching on pass two."""
        if way.tags.get('highway') == 'motorway_link':
            self.motorway_link_node_ids.update(node.ref for node in way.nodes)

    def relation(self, relation):
        tags = dict(relation.tags)
        if tags.get('type') != 'route' or tags.get('route') not in RAIL_ROUTE_TYPES | BUS_ROUTE_TYPES:
            return
        members = [(m.ref,m.role) for m in relation.members if m.type == 'w'
                   and not m.role.startswith(('platform','stop'))]
        nested = sum(m.type == 'r' for m in relation.members)
        self.audit['unsupported_nested_route_members'] += nested
        self.routes[relation.id] = tags,members,nested
        self.route_way_ids.update(ref for ref,_ in members)


def motorway_connector_features(connector_nodes, incident, surface_roads=SURFACE_ROADS):
    """Shared-node topology, never coordinate matching or ramp traversal.

    Each edge is an adjacent OSM node pair. Interior way vertices count too.
    Identical coordinates on different node IDs do not create a connection.
    The shared node does not need a ``highway=motorway_junction`` tag.
    """
    result = []
    for node_id, feature in sorted(connector_nodes.items()):
        edges = incident.get(node_id, [])
        links = [e for e in edges if e['highway'] == 'motorway_link']
        roads = [e for e in edges if e['highway'] in surface_roads]
        if not links or not roads:
            continue
        tags = feature['properties']
        result.append({'type':'Feature','geometry':feature['geometry'], 'properties':{
            'kind':'motorway_connector','category':'motorway_connector',
            'osmId':str(node_id),'osmType':'node','highway':tags.get('highway',''),
            'taggedMotorwayJunction':tags.get('highway') == 'motorway_junction',
            'name':tags.get('name',''),'ref':tags.get('ref',''),
            'connectedRoadTypes':sorted({e['highway'] for e in roads}),
            'motorwayLinkWayIds':sorted({str(e['wayId']) for e in links}),
            'connectedRoadWayIds':sorted({str(e['wayId']) for e in roads}),
            'incidentEdges':edges,
            'connectionRule':'shared_node_direct_incidence',
        }})
    return result


class Transport(osmium.SimpleHandler):
    def __init__(self, catalogue, bounds, tolerance, surface_roads, major_roads):
        super().__init__()
        self.catalogue,self.bounds,self.tolerance = catalogue,bounds,tolerance
        self.surface_roads,self.major_roads = surface_roads,major_roads
        self.incident = {}
        self.connector_nodes = {}
        self.roads = []
        self.stations = list(catalogue.stations)
        self.route_ways = {}
        self.seen_route_ways = set()
        self.audit = Counter(catalogue.audit)

    def way(self, way):
        tags = dict(way.tags)
        highway = tags.get('highway')
        # Inspect original node IDs before simplification or clipping. Do not
        # reject actual shared nodes just because bridge/layer tags change.
        if highway == 'motorway_link' or highway in self.surface_roads:
            nodes = list(way.nodes)
            refs = [node.ref for node in nodes]
            for i,node in enumerate(nodes):
                node_id = node.ref
                if node_id not in self.catalogue.motorway_link_node_ids:
                    continue
                if not node.location.valid():
                    self.audit['invalid_connector_node_locations'] += 1
                    continue
                x,y = node.location.lon,node.location.lat
                w,s,e,n = self.bounds
                if not (w <= x <= e and s <= y <= n):
                    continue
                self.connector_nodes.setdefault(
                    node_id,
                    self.catalogue.junctions.get(node_id)
                    or raw_feature(Point(x,y),{},node_id,'node'),
                )
                for j in (i-1,i+1):
                    if 0 <= j < len(refs) and refs[j] != node_id:
                        self.incident.setdefault(node_id,[]).append({
                            'wayId':str(way.id),'neighborNodeId':str(refs[j]),'highway':highway})
        route_member = way.id in self.catalogue.route_way_ids
        station = is_rail(tags)
        if route_member:
            self.seen_route_ways.add(way.id)
        if not (route_member or station or highway in self.major_roads):
            return
        try:
            geom = clipped(geometry(way),self.bounds,0,'line')
        except (osmium.InvalidLocationError, RuntimeError) as exc:
            self.audit['invalid_way_geometries'] += 1
            return
        if geom is None:
            return
        if station:
            self.stations.append(raw_feature(geom,tags,way.id,'way'))
        simplified = geom.simplify(self.tolerance,preserve_topology=True)
        if highway in self.major_roads:
            self.roads.append(normalize_road(raw_feature(simplified,tags,way.id,'way')))
        if route_member:
            self.route_ways[way.id] = simplified

    def area(self, area):
        tags = dict(area.tags)
        if not is_rail(tags):
            return
        try:
            geom = clipped(geometry(area,area=True),self.bounds,0,'polygon')
        except (osmium.InvalidLocationError, RuntimeError):
            self.audit['invalid_station_areas'] += 1
            return
        if geom is not None:
            self.stations.append(raw_feature(geom,tags,area.orig_id(),'way' if area.from_way() else 'relation'))

    def artifacts(self):
        rail,bus = [],[]
        # Area replaces line representation of the same way; preserve distinct
        # OSM identities even when their names match (possible nearby platforms).
        stations = {(f['properties']['osm_type'],f['properties']['osm_id']):f for f in self.stations}
        # Preserve the previous rail policy: prefer a named station node
        # over a line/area with the same name and category. Bus supply remains
        # node-based, avoiding new polygon/platform double counts.
        rail_keys = set()
        ordered = sorted(stations.values(), key=lambda f: f['properties']['osm_type'] != 'node')
        for feature in ordered:
            props = feature['properties']
            if is_rail(props):
                station = normalize_station(feature,'rail')
                key = (station['properties']['name'].strip().casefold(),station['properties']['category'])
                if props['osm_type'] == 'node' or not key[0] or key not in rail_keys:
                    rail.append(station)
                    if key[0]:
                        rail_keys.add(key)
            if props['osm_type'] == 'node' and is_bus(props):
                bus.append(normalize_station(feature,'bus'))
        for relation_id,(tags,members,nested) in sorted(self.catalogue.routes.items()):
            parts = []
            missing = sum(ref not in self.seen_route_ways for ref,_ in members)
            self.audit['missing_route_way_references'] += missing
            for ref,role in members:
                geom = self.route_ways.get(ref)
                if geom is None:
                    continue
                lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
                for line in lines:
                    coords = list(line.coords)
                    parts.append(coords[::-1] if role == 'backward' else coords)
            if not parts:
                continue
            feature = normalize_route(raw_feature(MultiLineString(parts),tags,relation_id,'relation'))
            feature['properties'].update(missingWayCount=missing,unsupportedNestedMemberCount=nested)
            (bus if tags['route'] in BUS_ROUTE_TYPES else rail).append(feature)
        connectors = motorway_connector_features(
            self.connector_nodes,self.incident,self.surface_roads)
        self.audit['tagged_junctions_in_bounds'] = len(self.catalogue.junctions)
        self.audit['motorway_link_nodes_in_bounds'] = len(self.connector_nodes)
        self.audit['qualified_connectors'] = len(connectors)
        self.audit['tagged_qualified_connectors'] = sum(
            feature['properties']['taggedMotorwayJunction'] for feature in connectors)
        self.audit['motorway_link_nodes_without_qualifying_road'] = sum(
            not any(e['highway'] in self.surface_roads for e in self.incident.get(n,[]))
            for n in self.connector_nodes)
        return {'king_county_rail.geojson':rail,'king_county_bus.geojson':bus,
                'king_county_major_roads.geojson':self.roads,
                'king_county_motorway_connectors.geojson':connectors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--deploy-dir',type=Path,default=DEFAULT_DEPLOY_DIR)
    add_extract_arguments(parser,Path(__file__).with_name('osm_transport_config.ini'))
    args = parser.parse_args()
    config = load_config(args.config)
    bounds,tolerance,_,_ = extraction_options(args,config)
    allowed = csv_set(config['transport']['connector_road_types'])
    if not allowed or not allowed <= SURFACE_ROADS:
        raise ValueError('connector_road_types must contain supported roads at trunk class or lower')
    catalogue = Catalogue(bounds)
    catalogue.apply_file(str(args.input))
    handler = Transport(catalogue,bounds,tolerance,allowed,csv_set(config['transport']['major_road_types']))
    handler.apply_file(str(args.input),locations=True,idx=args.location_index)
    artifacts = handler.artifacts()
    for filename,features in artifacts.items():
        path = args.output_dir / filename
        write_collection(path,features,source=args.input,bounds=bounds,
                         simplify_degrees=tolerance,audit=dict(handler.audit),metadata={
                             'connectorRule':'shared_node_direct_incidence',
                             'connectorRoadTypes':sorted(allowed)})
        if not args.no_deploy:
            deploy(path,args.deploy_dir / filename)
    print('Extraction audit:',json.dumps(handler.audit,sort_keys=True))
    if not artifacts['king_county_motorway_connectors.geojson']:
        print('WARNING: no shared motorway-link/surface-road nodes found; see extraction audit.')


if __name__ == '__main__':
    main()
