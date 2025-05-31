import os
import logging
import json
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_obj import StandardPythonParameterType
import app  # Import your FastAPI app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global fastapi_app
    fastapi_app = app.app
    logger.info("FastAPI app initialized")

@input_schema('data', StandardPythonParameterType({"messages": [{"role": "string", "content": "string"}]}))
@output_schema(StandardPythonParameterType({"response": "string"}))
def run(data):
    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)
    response = client.post("/generate", json=data)
    return response.json()
