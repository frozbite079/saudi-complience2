import weaviate
import weaviate.classes as wvc
import requests
import json

def test_rag():
    try:
        # 1. Generate Embedding via TEI
        tei_url = "http://localhost:8088/embed"
        query = "Electrical panel rooms should have signs or labels on doors"
        resp = requests.post(tei_url, json={"inputs": [query]})
        resp.raise_for_status()
        query_vector = resp.json()[0]

        # 2. Query Weaviate Database
        client = weaviate.connect_to_local(port=8081, grpc_port=50052)
        collection = client.collections.get("SBC_Rule")
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=3,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        results = []
        for obj in response.objects:
            obj_data = dict(obj.properties)
            obj_data['distance'] = obj.metadata.distance
            results.append(obj_data)
            
        client.close()
        
        with open('rag_test.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        with open('rag_test.json', 'w') as f:
            json.dump({"error": str(e)}, f, indent=2)

if __name__ == "__main__":
    test_rag()
