import weaviate

try:
    client = weaviate.connect_to_local(port=8081, grpc_port=50052)
    collection = client.collections.get("SBC_Rule")
    response = collection.query.fetch_objects(limit=3)
    
    with open("check_query_output.txt", "w") as f:
        if response.objects:
            f.write(f"Found {len(response.objects)} objects\n")
            for i, obj in enumerate(response.objects):
                f.write(f"--- Sample #{i+1} ---\n")
                f.write(f"Category: {obj.properties.get('category')}\n")
                f.write(f"Rule Text: {obj.properties.get('rule_text')}\n")
                f.write(f"SBC Reference: {obj.properties.get('sBC_reference')}\n")
                f.write(f"CV Target: {obj.properties.get('cV_target')}\n")
                f.write(f"Priority: {obj.properties.get('priority')}\n")
        else:
            f.write("No objects found\n")
    client.close()
    print("Exported query results.")
except Exception as e:
    with open("check_query_output.txt", "w") as f:
        f.write(f"Error: {str(e)}\n")
