import pandas as pd
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent

df = pd.read_csv(_PROJECT_ROOT / "data" / "DataSetTeensyv3.csv")
print(df['topic'].value_counts())
