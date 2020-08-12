# !pygmentize package/src/custom_lightgbm_inference/my_serving.py

from sagemaker_inference import model_server
from custom_lightgbm_inference import handler

HANDLER_SERVICE = handler.__name__

def main():
    print('Running handler service:', HANDLER_SERVICE)
    model_server.start_model_server(handler_service=HANDLER_SERVICE)
