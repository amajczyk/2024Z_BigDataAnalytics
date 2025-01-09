from cassandra.cluster import Cluster

def delete_ethereum_invalid_predictions():
    try:
        # Connect to Cassandra
        cluster = Cluster(["127.0.0.1"])
        session = cluster.connect()
        
        # Set keyspace
        session.set_keyspace("stream_predictions")
        
        # First, select all timestamps where prediction is -1 for ETHEREUM
        select_query = """
        SELECT symbol, timestamp 
        FROM model_predictions_10m 
        WHERE symbol = 'ETHEREUM' AND prediction = -1 
        ALLOW FILTERING
        """
        
        rows = session.execute(select_query)
        
        # Delete each record using the full primary key
        delete_count = 0
        for row in rows:
            delete_query = """
            DELETE FROM model_predictions_10m 
            WHERE symbol = %s AND timestamp = %s
            """
            session.execute(delete_query, (row.symbol, row.timestamp))
            delete_count += 1
            
        print(f"Successfully deleted {delete_count} records where symbol=ETHEREUM and prediction=-1")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
    finally:
        # Close the connections
        if cluster:
            cluster.shutdown()

if __name__ == "__main__":
    delete_ethereum_invalid_predictions()