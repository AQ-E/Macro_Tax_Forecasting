import pandas as pd
try:
    p1 = r"C:\Users\LENOVO\Downloads\Pakistan_Macro_Tax_Forecast\tax_prepared_data.xlsx"
    p2 = r"c:\Users\LENOVO\Downloads\Pakistan_Income_Tax_Slabs_app\Macro_Tax_Forecasting_Diag\Tax_forecasting.xlsx"
    
    df1 = pd.read_excel(p1, sheet_name="tax_prepared_data")
    df2 = pd.read_excel(p2, sheet_name="Sheet1")
    
    print(f"Data 1 (User) Columns: {df1.columns.tolist()[:5]}... Size: {df1.shape}")
    print(f"Data 2 (Training) Columns: {df2.columns.tolist()[:5]}... Size: {df2.shape}")
    
    # Check GST 2026 in both
    v1 = df1[df1['year_end'] == 2026]['gst'].values[0] if 'year_end' in df1.columns and 2026 in df1['year_end'].values else "N/A"
    # df2 might have 'Year' as column
    year_col = next((c for c in df2.columns if 'year' in str(c).lower()), None)
    gst_col = next((c for c in df2.columns if 'gst' in str(c).lower()), None)
    v2 = "N/A"
    if year_col and gst_col:
        v2 = df2[df2[year_col].astype(str).str.contains('2026')][gst_col].values[0] if not df2[df2[year_col].astype(str).str.contains('2026')].empty else "N/A"
        
    print(f"GST 2026 User: {v1}")
    print(f"GST 2026 Training: {v2}")

except Exception as e:
    print(f"Error: {e}")
