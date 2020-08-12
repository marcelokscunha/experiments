ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

# cd ../package/ && python setup.py sdist && cp dist/custom_lightgbm_framework-1.0.0.tar.gz ../docker/code/

echo "Building docker image..."
docker build -f docker/Dockerfile -t $REPO_NAME ./docker

echo "Tagging docker image..."
docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

$(aws ecr get-login --no-include-email --registry-ids $ACCOUNT_ID)

aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

echo "Pushing docker image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
