import weaviate

# Connect to Weaviate
print("Connecting to Weaviate...")
client = weaviate.connect_to_local(port=8081, grpc_port=50052)

try:
    # Get the collection
    collection = client.collections.get("SBC_Rule")

    # Query a sample rule and inspect its structure
    print("Querying sample rules...")
    response = collection.query.fetch_objects(limit=3)

    if response.objects:
        print(f"Found {len(response.objects)} objects in the collection")
        # Display the first result
        for i, obj in enumerate(response.objects):
            print(f"\n--- Sample Rule #{i+1} ---")
            print(f"Category: {obj.properties['category']}")
            print(f"Rule Text: {obj.properties['rule_text']}")
            print(f"SBC Reference: {obj.properties['sBC_reference']}")
            print(f"CV Target: {obj.properties['cV_target']}")
            print(f"Priority: {obj.properties['priority']}")
    else:
        print("No objects found in the collection")

    # Close the connection
    client.close()
    print("Connection closed.")
except Exception as e:
    print(f"Error: {e}")
    client.close()