import pandas as pd

def format_currency(val, unit='bn'):
    """Standardized currency formatter for Institutional reports."""
    if pd.isna(val): return "—"
    return f"{val:,.1f}"

def format_percent(val, decimals=1):
    """Standardized percentage formatter."""
    if pd.isna(val): return "—"
    return f"{val:.{decimals}f}%"

def get_table_style():
    """CSS for professional Institutional-grade tables."""
    return """
    <style>
    .report-table {
        font-family: 'Inter', sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .report-table td, .report-table th {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .report-table tr:nth-child(even){background-color: #f2f2f2;}
    .report-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: left;
        background-color: #2c3e50;
        color: white;
    }
    </style>
    """
