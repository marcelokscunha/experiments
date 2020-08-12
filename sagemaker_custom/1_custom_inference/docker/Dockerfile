# !pygmentize docker/Dockerfile

# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container
FROM sagemaker-training-containers/framework-container:latest

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6


# Framework Training docker
# COPY code/custom_lightgbm_framework-1.0.0.tar.gz /custom_lightgbm_framework-1.0.0.tar.gz

# Installing numpy, pandas, scikit-learn, scipy
# RUN ${PIP} install --no-cache --upgrade \
#         /custom_lightgbm_framework-1.0.0.tar.gz && \
#     rm /custom_lightgbm_framework-1.0.0.tar.gz

# Setting some environment variables.
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
#     PYTHONIOENCODING=UTF-8 \
#     LANG=C.UTF-8 \
#     LC_ALL=C.UTF-8


# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


# Previous inference docker
RUN apt-get update -y && apt-get -y install --no-install-recommends default-jdk
RUN rm -rf /var/lib/apt/lists/*

# RUN pip --no-cache-dir install multi-model-server sagemaker-inference sagemaker-training
# RUN pip --no-cache-dir install pandas numpy scipy scikit-learn

COPY code/custom_lightgbm_inference-0.1.0.tar.gz /custom_lightgbm_inference-0.1.0.tar.gz
        
RUN ${PIP} install --no-cache --upgrade \
        /custom_lightgbm_inference-0.1.0.tar.gz && \
    rm /custom_lightgbm_inference-0.1.0.tar.gz

# ENV PYTHONUNBUFFERED=TRUE
# ENV PYTHONDONTWRITEBYTECODE=TRUE
# ENV PYTHONPATH="/opt/ml/code:${PATH}"

#####################
# Required ENV vars #
#####################
# Set SageMaker training environment variables
# ENV SM_INPUT /opt/ml/input
# ENV SM_INPUT_TRAINING_CONFIG_FILE $SM_INPUT/config/hyperparameters.json
# ENV SM_INPUT_DATA_CONFIG_FILE $SM_INPUT/config/inputdataconfig.json
# ENV SM_CHECKPOINT_CONFIG_FILE $SM_INPUT/config/checkpointconfig.json

# Set SageMaker serving environment variables
ENV SM_MODEL_DIR /opt/ml/model

ENV CODE_DIR /opt/ml/code
# COPY main.py $CODE_DIR/main.py

# COPY src/my_serving.py $CODE_DIR/my_serving.py

# COPY src/handler_service.py $CODE_DIR/handler_service.py
# COPY src/inference_handler.py $CODE_DIR/inference_handler.py

# ENV SAGEMAKER_TRAINING_MODULE my_training:main

#Not injecting inference code, hence no need for env var
# ENV SAGEMAKER_SERVING_MODULE my_serving:main

# ENTRYPOINT ["python", "/opt/ml/code/main.py"]
