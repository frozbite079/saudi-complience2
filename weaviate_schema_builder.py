"""
Weaviate Schema Builder & Data Loader for SBC Compliance Rules
==============================================================
Reads the SBC401 Excel file, generates embeddings via TEI,
and populates Weaviate with rule data + custom vectors (BYOV).

Schema properties match the app/weaviate_client.py SCHEMA_PROPERTIES.
"""

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import pandas as pd
import requests
import warnings
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

TEI_URL = "http://localhost:8088/embed"
BATCH_SIZE = 8

SCHEMA_PROPERTIES = [
    Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="sub_category", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="rule_text", data_type=DataType.TEXT, skip_vectorization=False),
    Property(name="sbc_reference", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="cv_target", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="detection_type", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="priority", data_type=DataType.TEXT, skip_vectorization=True),
]


def get_embeddings_from_tei(texts: list) -> list:
    try:
        response = requests.post(TEI_URL, json={"inputs": texts}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching embeddings from TEI: {e}")
        raise


def init_weaviate_schema(client, collection_name="SBC_Rule"):
    print(f"Checking if collection '{collection_name}' exists...")
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it to recreate...")
        client.collections.delete(collection_name)

    print(f"Creating collection '{collection_name}'...")
    collection = client.collections.create(
        name=collection_name,
        description="Saudi Building Code (SBC) Rules for Vision-Detectable Compliance Analysis",
        properties=SCHEMA_PROPERTIES,
    )
    print(f"Collection '{collection_name}' created successfully with BYOV configuration!")
    return collection


def populate_database_from_excel(collection, excel_path: str):
    print(f"Loading data from {excel_path}...")

    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return

    try:
        xls = pd.ExcelFile(excel_path)
        print(f"Excel file loaded. Sheets found: {xls.sheet_names}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    total_inserted = 0

    for sheet_name in xls.sheet_names:
        raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        header_row_idx = None

        for idx, r in raw_df.head(20).iterrows():
            row_vals = [str(val).strip().lower() for val in r.values if pd.notna(val)]
            if "rule text" in row_vals or "rule_text" in row_vals or "sbc reference" in row_vals:
                header_row_idx = idx
                break

        if header_row_idx is None:
            print(f"Skipping sheet '{sheet_name}' - could not find header row.")
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row_idx)
        df.columns = [str(col).strip() for col in df.columns]

        has_rule_text = "Rule Text" in df.columns or "Rule_text" in df.columns
        has_detection = "Detection Type" in df.columns or "Detection_Type" in df.columns
        has_cv = "CV Target" in df.columns or "CV_target" in df.columns

        if not has_rule_text:
            print(f"Skipping sheet '{sheet_name}' - no Rule Text column found.")
            print(f"  Columns: {df.columns.tolist()}")
            continue

        print(f"Processing sheet '{sheet_name}' ({len(df)} rows)")
        print(f"  Detected columns: {df.columns.tolist()}")

        objects_to_insert = []
        texts_to_embed = []

        for _, row in df.iterrows():
            rule_text = str(row.get("Rule Text", row.get("Rule_text", ""))).strip()
            if not rule_text or rule_text.lower() == "nan":
                continue

            category_val = str(row.get("Category", "")).strip()
            if category_val.lower() == "nan" or not category_val:
                category_val = "Electrical"

            sbc_ref = str(row.get("SBC Reference", row.get("SBC_reference", ""))).strip()
            cv_target = str(row.get("CV Target", row.get("CV_target", ""))).strip()
            detection_type = str(row.get("Detection Type", row.get("Detection_Type", ""))).strip()
            priority = str(row.get("Priority", "")).strip()

            for field in [sbc_ref, cv_target, detection_type, priority]:
                if field.lower() == "nan":
                    field = ""

            item = {
                "category": "Electricity",
                "sub_category": category_val,
                "rule_text": rule_text,
                "sbc_reference": sbc_ref if sbc_ref.lower() != "nan" else "",
                "cv_target": cv_target if cv_target.lower() != "nan" else "",
                "detection_type": detection_type if detection_type.lower() != "nan" else "",
                "priority": priority if priority.lower() != "nan" else "",
            }

            objects_to_insert.append(item)
            texts_to_embed.append(rule_text)

        if objects_to_insert:
            print(f"  Found {len(objects_to_insert)} records. Generating embeddings via TEI...")

            all_embeddings = []
            for i in range(0, len(texts_to_embed), BATCH_SIZE):
                batch_texts = texts_to_embed[i : i + BATCH_SIZE]
                embeddings = get_embeddings_from_tei(batch_texts)
                all_embeddings.extend(embeddings)

            print(f"  Inserting {len(objects_to_insert)} records with custom vectors into Weaviate...")
            with collection.batch.dynamic() as batch:
                for data_row, custom_vector in zip(objects_to_insert, all_embeddings):
                    batch.add_object(properties=data_row, vector=custom_vector)

            total_inserted += len(objects_to_insert)
            print(f"  Processed {len(objects_to_insert)} records for '{sheet_name}'")

    xls.close()
    print(f"Successfully embedded and inserted {total_inserted} total rules into Weaviate.")


if __name__ == "__main__":
    print("Connecting to local Weaviate instance on port 8081...")
    try:
        client = weaviate.connect_to_local(port=8081, grpc_port=50052)
        try:
            collection = init_weaviate_schema(client)
            excel_path = "SBC401_VisionDetectableRules_VectorDB_2026-05-18.xlsx"
            populate_database_from_excel(collection, excel_path)
        finally:
            client.close()
    except Exception as e:
        print(f"Error connecting or processing Weaviate data: {e}")
        print("Please ensure your Weaviate instance is running locally on port 8081 (and gRPC on 50052).")
