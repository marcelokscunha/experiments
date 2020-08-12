import argparse
import os
import pandas as pd
import numpy as np
import logging
import sys

import joblib

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args):
    '''
    Main function for initializing SageMaker training in the hosted infrastructure.
    
    Parameters
    ----------
    args: the parsed input arguments of the script. The objects assigned as attributes of the namespace. It's the populated namespace.
    
    See: https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
    '''

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        print('0 len for input_files')
#         raise ValueError(('There are no files in {}.\n' +
#                           'This usually indicates that the channel ({}) was incorrectly specified,\n' +
#                           'the data specification in S3 was incorrectly specified or the role specified\n' +
#                           'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    data = pd.concat(raw_data)

    X=data.iloc[:,:4]
    y=data.iloc[:,4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    gbm = lgb.LGBMClassifier(objective='multiclass',
                            num_class=len(np.unique(y)))

    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_names='[validation_softmax]',
            eval_metric='softmax',
            early_stopping_rounds=5,
            verbose=5)


    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    score = f1_score(y_test,y_pred,labels=[0.0,1.0,2.0],average='micro')

    # generate evaluation metrics
    logger.info(f'[F1 score] {score}')
                                                              
    save_model(gbm, args.model_dir)
                                                              
def save_model(model, model_dir):
    '''
    Function for saving the model in the expected directory for SageMaker.
    
    Parameters
    ----------
    model: a Scikit-Learn estimator
    model_dir: A string that represents the path where the training job writes the model artifacts to. After training, artifacts in this directory are uploaded to S3 for model hosting. (this should be the default SageMaker environment variables)
    '''
    logger.info("Saving the model.")
                                                              
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    estimator = joblib.load(os.path.join(model_dir, "model.joblib"))
    return estimator


# Main script entry for SageMaker to run when initializing training
                                                              
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--MY-HYPERPARM-NAME', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
                                                              
    train(args)