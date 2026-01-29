import pandas as pd
df = pd.read_csv("DataSetTeensyv3.csv")
print(df['topic'].value_counts())