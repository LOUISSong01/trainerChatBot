import polars as pl
SAMPLE_DATA_PATH = "data/news_data.csv"
m_data = pl.read_csv(SAMPLE_DATA_PATH)

print(m_data.head())



