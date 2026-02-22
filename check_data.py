import pandas as pd
import sys

try:
    path = r"C:\Users\LENOVO\Downloads\Pakistan_Macro_Tax_Forecast\tax_prepared_data.xlsx"
    df = pd.read_excel(path, sheet_name="tax_prepared_data")
    print(f"Columns: {df.columns.tolist()}")
    
    # Try common names
    year_col = next((c for c in df.columns if 'year' in str(c).lower()), None)
    gst_col = next((c for c in df.columns if 'gst' in str(c).lower()), None)
    
    if year_col and gst_col:
        mask = df[year_col].astype(str).str.contains('2026')
        result = df[mask][[year_col, gst_col]]
        print("--- GST DATA FOR 2026 ---")
        print(result)
        print("-------------------------")
    else:
        print(f"Could not find columns. Year: {year_col}, GST: {gst_col}")
except Exception as e:
    print(f"Error: {e}")
