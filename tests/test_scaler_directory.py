import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pricemodel.model_manager import modelmanager


class ScalerDirectoryTest(unittest.TestCase):
    def test_scaler_fit_creates_fresh_timestamped_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = modelmanager()
            manager.directory = str(Path(temporary) / "new" / "run")
            manager._PROPERTY_FEATURES = ["sqft"]
            manager._TIME_FEATURES = []
            manager._MARKET_FEATURES = []
            manager._TARGET_FEATURE = "log_price"
            manager._LOCAL_FEATURES = []
            manager.local_feature_dim = 0
            manager._scale_mode = "fit"
            manager.dataframe = pd.DataFrame({
                "sqft": [1000.0, 1200.0, 1400.0],
                "log_price": np.log([300000.0, 350000.0, 400000.0]),
            })
            manager._community_tensor = None
            manager._build_tensor_dataset = lambda: None

            manager._fit_and_apply_scalers_on_split(np.array([0, 1]))

            self.assertTrue(Path(manager.directory, "sqft_scaler.pkl").exists())
            self.assertTrue(Path(manager.directory, "log_price_scaler.pkl").exists())


if __name__ == "__main__":
    unittest.main()
