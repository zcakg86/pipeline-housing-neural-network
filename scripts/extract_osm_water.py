#!/usr/bin/env python3
"""Extract buffered King County water polygons and coastline using pyosmium."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import osmium

if __package__:
    from .osm_common import (add_extract_arguments, load_config, extraction_options, csv_set, geometry, clipped, raw_feature, write_collection, deploy)
else:
    from osm_common import (add_extract_arguments, load_config, extraction_options, csv_set, geometry, clipped, raw_feature, write_collection, deploy)

DEFAULT_INPUT = Path('data/osm/washington-latest.osm.pbf')
DEFAULT_OUTPUT = Path('data/osm/king_county_water.geojson')
DEFAULT_DEPLOY_OUTPUT = Path('java-app/house-price-app/src/main/resources/META-INF/resources/data/king_county_water.geojson')


class Water(osmium.SimpleHandler):
    def __init__(self,bounds,tolerance,excluded,exclude_waterway=True):
        super().__init__()
        self.bounds,self.tolerance = bounds,tolerance
        self.excluded,self.exclude_waterway = excluded,exclude_waterway
        self.features = []
        self.audit = Counter()

    def add(self,obj,tags,area):
        try:
            geom = clipped(geometry(obj,area=area),self.bounds,self.tolerance,
                           'polygon' if area else 'line')
        except (osmium.InvalidLocationError,RuntimeError):
            self.audit['invalid_geometries'] += 1
            return
        if geom is None:
            return
        osm_id = obj.orig_id() if area else obj.id
        osm_type = ('way' if obj.from_way() else 'relation') if area else 'way'
        feature = raw_feature(geom,tags,osm_id,osm_type)
        coastline = tags.get('natural') == 'coastline'
        feature['properties'] = {
            'osmId':str(osm_id),'osmType':osm_type,'name':tags.get('name',''),
            'category':'coastline' if coastline else 'natural_water',
            'type':'coastline' if coastline else tags.get('water') or tags.get('waterway') or 'water',
            'natural':tags.get('natural',''),'waterway':tags.get('waterway',''),
            'intermittent':tags.get('intermittent','')}
        self.features.append(feature)

    def way(self,way):
        tags = dict(way.tags)
        if tags.get('natural') == 'coastline':
            self.add(way,tags,False)

    def area(self,area):
        tags = dict(area.tags)
        if (tags.get('natural') == 'water' and tags.get('water') not in self.excluded
                and not (self.exclude_waterway and tags.get('waterway'))):
            self.add(area,tags,True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--deploy-output',type=Path,default=DEFAULT_DEPLOY_OUTPUT)
    add_extract_arguments(parser,Path(__file__).with_name('osm_water_config.ini'))
    args = parser.parse_args()
    config = load_config(args.config)
    bounds,tolerance,original,buffer = extraction_options(args,config)
    handler = Water(bounds,tolerance,csv_set(config['water']['excluded_water_types']),
                    config['water'].getboolean('exclude_waterway_tagged',True))
    handler.apply_file(str(args.input),locations=True,idx=args.location_index)
    write_collection(args.output,handler.features,source=args.input,bounds=bounds,
                     simplify_degrees=tolerance,audit=dict(handler.audit),metadata={
                         'kingCountyBounds':list(original) if original else None,
                         'bufferDegrees':buffer if original else 0})
    if not args.no_deploy:
        deploy(args.output,args.deploy_output)
    print('Extraction audit:',json.dumps(handler.audit,sort_keys=True))


if __name__ == '__main__':
    main()
