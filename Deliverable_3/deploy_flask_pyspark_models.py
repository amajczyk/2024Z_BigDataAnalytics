from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
import os
from flask import Flask, request, jsonify

# Initialize Spark Session
spark = SparkSession.builder.appName("DeployLatestModels").getOrCreate()

# Directory containing model files
models_dir = os.path.join(os.getcwd(), "models")  # Ensure absolute path to models directory

# Symbol-specific feature definitions
symbols_features = {
    "BP": ["volume", "volatility", "market_sentiment", "trading_activity", 
            "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "COP": ["volume", "volatility", "market_sentiment", "trading_activity", 
             "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "SHEL": ["volume", "volatility", "market_sentiment", "trading_activity", 
              "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "XOM": ["volume", "volatility", "market_sentiment", "trading_activity", 
             "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "ETHEREUM": ["bid", "ask", "spread_raw", "spread_table", 
                  "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"]
}

def create_flask_app(symbol):
    app = Flask(__name__)

    # Keep track of deployed models
    deployed_models = {}

    @app.route(f'/{symbol}/deploy', methods=['POST'])
    def deploy_model_symbol():
        """
        Endpoint to deploy a model for the specific symbol.
        """
        latest_models = get_latest_models(models_dir)
        if symbol not in latest_models:
            return jsonify({"error": f"No models found for symbol {symbol}"}), 404

        model_file = latest_models[symbol]["file"]
        model_path = os.path.join(models_dir, model_file)

        # Load the PySpark model
        try:
            model = PipelineModel.load(model_path)
            deployed_models[symbol] = {"path": model_path, "model": model}
            return jsonify({"message": f"Successfully deployed {model_file} for {symbol}"}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    @app.route(f'/{symbol}/status', methods=['GET'])
    def status_symbol():
        """
        Endpoint to check the status of deployed models for the specific symbol.
        """
        status = {key: value["path"] for key, value in deployed_models.items()}
        return jsonify(status), 200

    @app.route(f'/{symbol}/serve', methods=['POST'])
    def serve_model_symbol():
        """
        Endpoint to serve a deployed model for the specific symbol.
        """
        if symbol not in deployed_models:
            return jsonify({"error": f"No deployed model found for {symbol}"}), 404

        model = deployed_models[symbol]["model"]
        input_data = request.get_json()

        # Ensure input_data is a list
        if not isinstance(input_data, list):
            return jsonify({"error": "Input data must be a list of records."}), 400

        # Convert input_data to a Spark DataFrame
        try:
            spark_df = spark.createDataFrame(input_data)

            # Add symbol-specific feature engineering if required
            if symbol in symbols_features:
                feature_col_name = f"{symbol}_features"
                assembler = VectorAssembler(inputCols=symbols_features[symbol], outputCol=feature_col_name)
                spark_df = assembler.transform(spark_df)

            predictions = model.transform(spark_df)
            predictions = predictions.toPandas()

            # Convert DenseVector or other non-serializable types to JSON serializable
            for col in predictions.columns:
                predictions[col] = predictions[col].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )

            result = predictions.to_dict(orient="records")
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Failed to serve model: {str(e)}"}), 500


    return app

def get_latest_models(models_dir):
    """
    Identify the latest model for each symbol in the directory.
    """
    model_files = os.listdir(models_dir)
    model_dict = {}

    for model_file in model_files:
        # Skip files that don't match the naming convention
        if not model_file.startswith(("BP_", "COP_", "ETHEREUM_", "SHEL_", "XOM_")):
            continue

        # Extract the symbol and timestamp
        parts = model_file.split("_")
        symbol = parts[0]  # Extract the symbol (e.g., BP, COP, etc.)
        timestamp = parts[-2] + parts[-1]  # Combine timestamp parts

        # Update the latest model for the symbol
        if symbol not in model_dict or timestamp > model_dict[symbol]["timestamp"]:
            model_dict[symbol] = {"file": model_file, "timestamp": timestamp}

    return model_dict

if __name__ == '__main__':
    import threading

    # Define ports for each symbol
    symbols_ports = {
        "BP": 5000,
        "COP": 5001,
        "ETHEREUM": 5002,
        "SHEL": 5003,
        "XOM": 5004
    }

    threads = []

    for symbol, port in symbols_ports.items():
        app = create_flask_app(symbol)
        thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": port})
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
