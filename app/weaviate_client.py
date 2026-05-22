from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter, MetadataQuery

from app.config import (
    WEAVIATE_HOST,
    WEAVIATE_PORT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_COLLECTION,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

ALLOWED_CLASSIFICATIONS = {
    "Structural Safety",
    "Electrical",
    "Electricity",
    "Plumbing",
    "Fire Safety",
}

CLASSIFICATION_TO_DB_CATEGORY = {
    "Electrical": "Electricity",
}

SCHEMA_PROPERTIES = [
    Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="sub_category", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="rule_text", data_type=DataType.TEXT, skip_vectorization=False),
    Property(name="sbc_reference", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="cv_target", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="detection_type", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="priority", data_type=DataType.TEXT, skip_vectorization=True),
]


@contextmanager
def weaviate_connection() -> Generator[weaviate.WeaviateClient, None, None]:
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    try:
        yield client
    finally:
        client.close()


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    if not client.collections.exists(WEAVIATE_COLLECTION):
        client.collections.create(
            name=WEAVIATE_COLLECTION,
            description="Saudi Building Code (SBC) Rules for Vision-Detectable Compliance Analysis",
            properties=SCHEMA_PROPERTIES,
        )
        logger.info("Collection '%s' created.", WEAVIATE_COLLECTION)


def insert_rules_batch(
    client: weaviate.WeaviateClient,
    objects: list[dict],
    vectors: list[list[float]],
) -> int:
    collection = client.collections.get(WEAVIATE_COLLECTION)
    inserted = 0
    with collection.batch.dynamic() as batch:
        for props, vec in zip(objects, vectors):
            batch.add_object(properties=props, vector=vec)
            inserted += 1
    return inserted


def search_rules(
    query_vector: list[float],
    top_k: int | None = None,
    classification: str | None = None,
) -> list[dict]:
    limit = top_k or RAG_TOP_K
    where_filter = None
    if classification:
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of: {', '.join(sorted(ALLOWED_CLASSIFICATIONS))}"
            )
        db_category = CLASSIFICATION_TO_DB_CATEGORY.get(classification, classification)
        where_filter = Filter.by_property("category").equal(db_category)

    try:
        with weaviate_connection() as client:
            collection = client.collections.get(WEAVIATE_COLLECTION)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=where_filter,
                return_metadata=MetadataQuery(distance=True),
            )
            results = []
            for obj in response.objects:
                results.append(
                    {
                        "category": obj.properties.get("category", ""),
                        "sub_category": obj.properties.get("sub_category", ""),
                        "rule_text": obj.properties.get("rule_text", ""),
                        "sbc_reference": obj.properties.get("sbc_reference", ""),
                        "cv_target": obj.properties.get("cv_target", ""),
                        "detection_type": obj.properties.get("detection_type", ""),
                        "priority": obj.properties.get("priority", ""),
                        "distance": obj.metadata.distance,
                    }
                )
            return results
    except Exception:
        logger.exception("Weaviate search failed")
        return []


def get_collection_stats(client: weaviate.WeaviateClient) -> dict:
    collection = client.collections.get(WEAVIATE_COLLECTION)
    total = collection.aggregate.over_all(total_count=True).total_count
    return {"collection": WEAVIATE_COLLECTION, "total_objects": total}
