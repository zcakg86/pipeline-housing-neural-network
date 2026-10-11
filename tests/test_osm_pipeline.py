"""Integration checks run against a generated, topology-controlled OSM fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from scripts.extract_osm_transport import Catalogue, Transport, MAJOR_ROAD_TYPES
from scripts.extract_osm_water import Water
from scripts.osm_common import SURFACE_ROADS, Boundary, extraction_options, load_config
from scripts.build_transport_features import build_features
from shapely.geometry import shape
import osmium

BOUNDS = (-122.5,47.3,-122.1,47.8)


def fixture(path):
    root = ET.Element('osm',version='0.6',generator='test')
    ways, relations = [],[]
    def tags(obj, values):
        for k,v in values.items():
            ET.SubElement(obj,'tag',k=k,v=v)
    def node(i,x=None,y=None,**values):
        obj = ET.SubElement(root,'node',id=str(i),version='1',lon=str(x if x is not None else -122.4+i*0.0001),lat=str(y if y is not None else 47.5))
        tags(obj,values)
    def way(i,refs,**values):
        ways.append((i,refs,values))
    # A positive junction is an interior vertex of the link way.
    for i in range(1,31):
        node(i,**({'highway':'motorway_junction'} if i in {1,4,10,13,16,19} else {}))
    way(101,[2,1,3],highway='motorway_link')
    way(102,[1,21],highway='residential')
    way(103,[4,5],highway='motorway_link')
    way(104,[4,6],highway='motorway')
    way(105,[7,8],highway='motorway_link')
    way(106,[7,9],highway='primary')  # no tag on node 7
    way(107,[10,11],highway='motorway_link')
    way(108,[10,12],highway='footway')
    way(109,[13,14],highway='motorway_link')
    node(31,-122.4+13*0.0001,47.5)  # same coords, distinct node ID
    way(110,[31,15],highway='trunk')
    way(111,[16,17],highway='motorway_link',bridge='yes',layer='1')
    way(112,[16,18],highway='trunk',layer='0')  # real shared node wins
    way(113,[19,20],highway='secondary')
    node(40,-122.35,47.55,railway='station',station='light_rail',name='Test rail')
    node(41,-122.35,47.551,highway='bus_stop',route_ref='10;20')
    # Closed water way.
    for i,(x,y) in enumerate([(-122.30,47.6),(-122.29,47.6),(-122.29,47.61),(-122.30,47.61)],50):
        node(i,x,y)
    way(150,[50,51,52,53,50],natural='water',water='lake')
    # Multipolygon outer and inner rings, also used for the boundary outer.
    for i,(x,y) in enumerate([(-122.45,47.4),(-122.15,47.4),(-122.15,47.7),(-122.45,47.7),
                              (-122.25,47.62),(-122.20,47.62),(-122.20,47.65),(-122.25,47.65)],60):
        node(i,x,y)
    way(160,[60,61,62,63,60])
    way(161,[64,65,66,67,64])
    way(170,[21,22],natural='coastline')
    for i,refs,values in ways:
        obj = ET.SubElement(root,'way',id=str(i),version='1')
        for ref in refs:
            ET.SubElement(obj,'nd',ref=str(ref))
        tags(obj,values)
    for i, members, values in [
        (200,[('way',101,''),('way',102,'backward'),('way',999,'')],{'type':'route','route':'bus','ref':'10'}),
        (201,[('way',160,'outer'),('way',161,'inner')],{'type':'multipolygon','natural':'water','water':'lake'}),
        (202,[('way',160,'outer')],{'type':'boundary','boundary':'administrative','admin_level':'6','name':'King County'}),
    ]:
        obj = ET.SubElement(root,'relation',id=str(i),version='1')
        for typ,ref,role in members:
            ET.SubElement(obj,'member',type=typ,ref=str(ref),role=role)
        tags(obj,values)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'sample.osm'
        fixture(self.source)
        # Exercise the actual binary PBF reader, not only OSM XML.
        class Copy(osmium.SimpleHandler):
            def node(self, obj): writer.add_node(obj)
            def way(self, obj): writer.add_way(obj)
            def relation(self, obj): writer.add_relation(obj)
        pbf = self.root/'sample.osm.pbf'
        with osmium.SimpleWriter(str(pbf)) as writer:
            Copy().apply_file(str(self.source))
        self.source = pbf

    def transport(self):
        catalogue = Catalogue(BOUNDS)
        catalogue.apply_file(str(self.source))
        handler = Transport(catalogue,BOUNDS,0.00008,SURFACE_ROADS,MAJOR_ROAD_TYPES)
        handler.apply_file(str(self.source),locations=True)
        return handler,handler.artifacts()

    def test_configured_clip_bounds_avoid_boundary_lookup(self):
        config_path = self.root/'extract.ini'
        config_path.write_text(
            '[extract]\nclip_bounds = -122.6,47.0,-121.0,47.9\n'
            'buffer_degrees = 0.05\nsimplify_degrees = 0.00008\n'
        )
        args = SimpleNamespace(
            input=self.source, bbox=None, buffer_degrees=None,
            simplify_degrees=None, location_index='flex_mem',
        )
        bounds,tolerance,original,buffer = extraction_options(
            args,load_config(config_path))
        self.assertEqual(bounds,(-122.6,47.0,-121.0,47.9))
        self.assertEqual(tolerance,0.00008)
        self.assertIsNone(original)
        self.assertEqual(buffer,0.05)

    def test_shared_node_topology_does_not_require_junction_tag(self):
        handler,artifacts = self.transport()
        features = artifacts['king_county_motorway_connectors.geojson']
        self.assertEqual({f['properties']['osmId'] for f in features},{'1','7','16'})
        self.assertEqual(handler.audit['tagged_junctions_in_bounds'],6)
        self.assertEqual(handler.audit['qualified_connectors'],3)
        self.assertEqual(handler.audit['tagged_qualified_connectors'],2)
        untagged = next(f for f in features if f['properties']['osmId'] == '7')
        self.assertFalse(untagged['properties']['taggedMotorwayJunction'])
        self.assertEqual(untagged['properties']['connectedRoadTypes'],['primary'])
        self.assertEqual(features[0]['properties']['motorwayLinkWayIds'],['101'])
        self.assertEqual(len(features[0]['properties']['incidentEdges']),3)

    def test_routes_stations_and_missing_members(self):
        handler,artifacts = self.transport()
        routes = [f for f in artifacts['king_county_bus.geojson'] if f['properties']['kind']=='route']
        self.assertEqual(len(routes),1)
        self.assertEqual(routes[0]['properties']['missingWayCount'],1)
        self.assertEqual(routes[0]['geometry']['type'],'MultiLineString')
        self.assertEqual(artifacts['king_county_rail.geojson'][0]['properties']['category'],'light_rail')

    def test_water_areas_and_holes(self):
        handler = Water(BOUNDS,0.00003,{'stream','drain','ditch','weir'})
        handler.apply_file(str(self.source),locations=True)
        identities = {(f['properties']['osmType'],f['properties']['osmId']) for f in handler.features}
        self.assertEqual(identities,{('way','150'),('relation','201'),('way','170')})
        lake = next(shape(f['geometry']) for f in handler.features if f['properties']['osmId']=='201')
        self.assertEqual(sum(len(p.interiors) for p in lake.geoms) if lake.geom_type=='MultiPolygon' else len(lake.interiors),1)

    def test_boundary_assembly(self):
        boundary = Boundary('King County','6')
        boundary.apply_file(str(self.source),locations=True)
        self.assertEqual(boundary.matches[202],(-122.45,47.4,-122.15,47.7))

    def test_feature_contract_empty_layers_and_projection(self):
        handler,artifacts = self.transport()
        sources = []
        for name in ('rail','bus','major_roads','motorway_connectors'):
            path = self.root/f'{name}.geojson'
            path.write_text(json.dumps({'type':'FeatureCollection','features':artifacts[f'king_county_{name}.geojson']}))
            sources.append(path)
        grid = self.root/'grid.json'
        grid.write_text(json.dumps({'cells':[{'h3_l8':'fixture','lng':-122.35,'lat':47.55}]}))
        result = build_features(grid,*sources)
        self.assertEqual(result['schema_version'],2)
        self.assertEqual(result['cells']['fixture']['light_rail_proximity'],1.0)
        self.assertEqual(result['cells']['fixture']['nearby_bus_route_count'],2)
        self.assertEqual(result['parameters']['metric_crs'],'EPSG:32610')
        self.assertEqual(len(result['cells']['fixture']),7)
        for path in sources:
            path.write_text('{"type":"FeatureCollection","features":[]}')
        result = build_features(grid,*sources)
        self.assertTrue(all(v == 0 for v in result['cells']['fixture'].values()))

    def test_cli_end_to_end(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        common = ['--input',str(self.source),'--no-deploy']
        for name,output_args in [('extract_osm_transport.py',['--output-dir',str(self.root)]),
                                 ('extract_osm_water.py',['--output',str(self.root/'water.geojson')])]:
            subprocess.run([sys.executable,str(scripts/name),*common,*output_args],check=True,capture_output=True,text=True)
        self.assertTrue((self.root/'water.geojson').exists())
        self.assertTrue((self.root/'king_county_motorway_connectors.geojson').exists())


if __name__ == '__main__':
    unittest.main()
