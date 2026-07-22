import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.pricemodel.water_features import add_water_proximity_features


class WaterFeatureTest(unittest.TestCase):
    def test_distance_is_to_polygon_boundary_and_threshold_is_inclusive(self):
        geometry = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.01, 47.39], [-122.00, 47.39],
                        [-122.00, 47.40], [-122.01, 47.40],
                        [-122.01, 47.39],
                    ]],
                },
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "water.geojson"
            path.write_text(json.dumps(geometry))
            result = add_water_proximity_features(
                pd.DataFrame({"lat": [47.395], "lng": [-122.005]}),
                water_path=path,
                cache_dir=None,
            )

        # The point lies inside the polygon, but is hundreds of metres from
        # its boundary. Polygon-interior distance would incorrectly be zero.
        self.assertGreater(result.loc[0, "distance_to_water_m"], 300.0)
        self.assertAlmostEqual(
            result.loc[0, "water_proximity"],
            np.exp(-result.loc[0, "distance_to_water_m"] / 100.0),
            places=7,
        )
        self.assertEqual(result.loc[0, "is_waterfront"], 0.0)


if __name__ == "__main__":
    unittest.main()
