import numpy as np
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
import duckdb
import pandasai

print("All imports successful")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Streamlit version: {st.__version__}")
print(f"PandasAI version: {pandasai.__version__.__version__}") 
print(f"DuckDB version: {duckdb.__version__}")