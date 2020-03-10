if [ -z "$1" ]; then
  echo "Need to supply 2 arguments: experiment_name & run_name"
  return
fi

if [ -z "$2" ]; then
  echo "Need to supply 2 arguments: experiment_name & run_name"
  return
fi

EXPERIMENT_NAME=$1
RUN_NAME=$2
echo 'EXPERIMENT_NAME:' $EXPERIMENT_NAME
echo 'RUN_NAME:       ' $RUN_NAME

. setup_environment_variables.sh
MLFLOW_TRACKING_URI=$DIR_MLFLOW mlflow experiments create -n $EXPERIMENT_NAME
MLFLOW_TRACKING_URI=$DIR_MLFLOW mlflow run . -e main -P experiment_name=$EXPERIMENT_NAME -P run_name=$RUN_NAME --no-conda