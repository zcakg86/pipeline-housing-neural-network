import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.extract_osm_transport import (
    classify_rail_station,
    motorway_connector_features,
    normalize_road,
    representative_point,
    route_list,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_DATA = (
    ROOT / "java-app/house-price-app/src/main/resources/META-INF/resources/data"
)


class OsmTransportTest(unittest.TestCase):
    def test_transport_classification_helpers(self):
        self.assertEqual(
            classify_rail_station({"railway": "station", "light_rail": "yes"}),
            "light_rail",
        )
        self.assertEqual(
            classify_rail_station({"railway": "tram_stop", "tram": "yes"}),
            "tram",
        )
        self.assertEqual(route_list("40; 256;C Line"), ["40", "256", "C Line"])
        road = normalize_road({
            "geometry": {"type": "LineString", "coordinates": []},
            "properties": {"osm_id": "1", "highway": "motorway_link"},
        })
        self.assertEqual(road["properties"]["category"], "motorway")
        self.assertTrue(road["properties"]["link"])
        point = representative_point({
            "type": "LineString",
            "coordinates": [[-122.4, 47.5], [-122.2, 47.7], [-122.4, 47.5]],
        })
        self.assertEqual(point["type"], "Point")
        self.assertAlmostEqual(point["coordinates"][0], -122.3)
        self.assertAlmostEqual(point["coordinates"][1], 47.6)

    def test_motorway_connector_requires_shared_osm_node_not_geometry_overlap(self):
        link = {
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
            "properties": {"osm_id": "link", "highway": "motorway_link"},
        }
        crossing = {
            "geometry": {"type": "LineString", "coordinates": [[0.5, -1], [0.5, 1]]},
            "properties": {"highway": "primary"},
        }
        different_level = {
            "geometry": {"type": "LineString", "coordinates": [[1, 0], [2, 0]]},
            "properties": {"highway": "primary", "layer": "1"},
        }
        touching = {
            "geometry": {"type": "LineString", "coordinates": [[1, 0], [2, 0]]},
            "properties": {"highway": "secondary"},
        }
        self.assertEqual(motorway_connector_features([link], [crossing]), [])
        self.assertEqual(motorway_connector_features([link], [different_level]), [])
        connectors = motorway_connector_features([link], [crossing, touching])
        self.assertEqual(len(connectors), 1)
        self.assertEqual(connectors[0]["geometry"]["coordinates"], [1, 0])
        self.assertEqual(
            connectors[0]["properties"]["connectedRoadTypes"], ["secondary"]
        )

    def test_deployed_transport_artifacts_are_self_consistent(self):
        expected = {
            "king_county_rail.geojson": {"station", "route"},
            "king_county_bus.geojson": {"station", "route"},
            "king_county_major_roads.geojson": {"road"},
            "king_county_motorway_connectors.geojson": {"motorway_connector"},
        }
        for filename, expected_kinds in expected.items():
            with self.subTest(filename=filename):
                payload = json.loads((STATIC_DATA / filename).read_text())
                features = payload["features"]
                metadata = payload["metadata"]
                self.assertEqual(metadata["featureCount"], len(features))
                self.assertEqual(
                    Counter(feature["properties"]["kind"] for feature in features),
                    Counter(metadata["countsByKind"]),
                )
                self.assertEqual(
                    {feature["properties"]["kind"] for feature in features},
                    expected_kinds,
                )
                self.assertTrue(all(feature.get("geometry") for feature in features))

        rail = json.loads((STATIC_DATA / "king_county_rail.geojson").read_text())
        rail_stations = [
            feature for feature in rail["features"]
            if feature["properties"]["kind"] == "station"
        ]
        self.assertTrue(all(
            feature["properties"]["osmId"] for feature in rail_stations
        ))
        judkins = [
            feature for feature in rail["features"]
            if feature["properties"]["osmId"] == "753224844"
        ]
        self.assertEqual(len(judkins), 1)
        self.assertEqual(judkins[0]["properties"]["category"], "light_rail")
        self.assertEqual(judkins[0]["geometry"]["type"], "Point")

    def test_frontend_exposes_all_transport_layer_controls(self):
        html = (
            ROOT
            / "java-app/house-price-app/src/main/resources/META-INF/resources/index.html"
        ).read_text()
        for control in ("showRail", "showBus", "showMajorRoads", "showTransportFeatures"):
            self.assertIn(f'id="{control}"', html)
        self.assertIn('src="js/layers/transport.js?', html)
        self.assertIn('src="js/layers/transport-features.js?', html)

    def test_deployed_h3_transport_features_match_grid(self):
        grid_path = (
            ROOT / "java-app/house-price-app/src/main/resources/model-artifacts/"
            "synthetic_h3_l8_grid.json"
        )
        artifact_path = (
            ROOT / "java-app/house-price-app/src/main/resources/model-artifacts/"
            "h3_l8_transport_features.json"
        )
        grid = json.loads(grid_path.read_text())
        artifact = json.loads(artifact_path.read_text())
        expected = {
            "light_rail_proximity",
            "bus_stop_density_per_sq_km",
            "nearby_bus_route_count",
            "bus_centrality_score",
            "motorway_connector_proximity",
            "major_road_accessibility",
            "major_road_density_km_per_sq_km",
        }
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(set(artifact["feature_names"]), expected)
        self.assertEqual(artifact["cell_count"], grid["cell_count"])
        self.assertEqual(
            set(artifact["cells"]), {cell["h3_l8"] for cell in grid["cells"]}
        )
        for values in artifact["cells"].values():
            self.assertEqual(set(values), expected)
            self.assertGreaterEqual(values["bus_centrality_score"], 0.0)
            self.assertLessEqual(values["bus_centrality_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
