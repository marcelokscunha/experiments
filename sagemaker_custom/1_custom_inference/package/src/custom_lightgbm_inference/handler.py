# !pygmentize package/src/custom_lightgbm_inference/handler.py
import os
import sys
import joblib
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference import content_types, errors, transformer, encoder, decoder

class HandlerService(DefaultHandlerService, DefaultInferenceHandler):
    def __init__(self):
        op = transformer.Transformer(default_inference_handler=self)
        super(HandlerService, self).__init__(transformer=op)
    
    ## Loads the model from the disk
    def default_model_fn(self, model_dir):
        model_filename = os.path.join(model_dir, "model.joblib")
        return joblib.load(model_filename)
    
    ## Parse and check the format of the input data
    def default_input_fn(self, input_data, content_type):
        if content_type != "text/csv":
            raise Exception("Invalid content-type: %s" % content_type)
        return decoder.decode(input_data, content_type).reshape(1,-1)
    
    ## Run our model and do the prediction
    def default_predict_fn(self, payload, model):
        return model.predict( payload ).tolist()
    
    ## Gets the prediction output and format it to be returned to the user
    def default_output_fn(self, prediction, accept):
        if accept != "text/csv":
            raise Exception("Invalid accept: %s" % accept)
        return encoder.encode(prediction, accept)
