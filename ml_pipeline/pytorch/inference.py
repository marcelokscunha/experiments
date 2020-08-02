import os
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from six import BytesIO

import boto3


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
        
def model_fn(model_dir):
    class Net(nn.Module):
        def __init__(self, hidden_channels, kernel_size, drop_out):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(hidden_channels, 20, kernel_size=kernel_size)
            self.conv2_drop = nn.Dropout2d(p=drop_out)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hard-coded apenas para exemplificar (poderiamos ter salvo os hyper-params no .pth)
    hidden_channels=10
    kernel_size=5
    dropout = 0.2
    
    model = torch.nn.DataParallel(Net(hidden_channels, kernel_size, dropout))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        return model.to(device)


def input_fn(input_data, request_content_type):
    logger.info('Pre-processing input data...')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if request_content_type=='application/x-npy':
        stream = BytesIO(input_data)
        np_array = np.load(stream, allow_pickle=True)
        tensor = torch.from_numpy(np_array)
    else:
        raise Exception(f'Unsupported request ContentType in request_content_type: {request_content_type}')
    
    return tensor.to(device)


def predict_fn(pre_processed_data, model):
    logger.info('Making predictions on pre-processed data...')
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        pre_processed_data = pre_processed_data.to(device)
        model.eval()
        output = model(pre_processed_data)
        
#         # Exemplo com acelerador de GPU anexado a máquina CPU (Elastic Inference)
#         device = torch.device("cpu")
#         model = model.to(device)
#         input_data = data.to(device)
#         model.eval()
#         with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
#             output = model(input_data)

    return output


def output_fn(prediction_output, response_content_type):
    logger.info('Pos-processing input data...')
    
    if type(prediction_output) == torch.Tensor:
        prediction_output = prediction_output.detach().cpu().numpy()
        
    if response_content_type == 'application/x-npy':
        buffer = BytesIO()
        np.save(buffer, prediction_output)
    else:
        raise Exception(f'Unsupported Accept in response_content_type: {response_content_type}')

        
    return buffer.getvalue()
   
