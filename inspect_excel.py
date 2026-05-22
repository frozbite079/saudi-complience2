import pandas as pd
import json

try:
    # Read without assuming row 0 is the header
    df = pd.read_excel('SBC401_VisionDetectableRules_VectorDB_2026-05-18.xlsx', sheet_name=0, header=None)
    
    # Get the first 15 rows to find the headers and some data
    rows = df.head(15).to_dict(orient='records')

    with open('columns.json', 'w') as f:
        json.dump(rows, f, default=str)
    print("Successfully dumped the first 15 rows to columns.json")
except Exception as e:
    print("Error:", str(e))
