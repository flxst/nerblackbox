
########################
# INPUT VERIFICATION
########################
if [[ -z "$1" ]]; then
  echo "Need to supply at least 1 argument: experiment_name"
  return
fi

########################
# INPUT PROCESSING
########################
EXPERIMENT_NAME=$1
RUN_NAME=$2
echo 'EXPERIMENT_NAME:' $EXPERIMENT_NAME
echo 'RUN_NAME:       ' $RUN_NAME

########################
# ENVIRONMENT VARIABLES
########################
. setup_environment_variables.sh

########################
# RUN EXPERIMENT
########################
MLFLOW_TRACKING_URI=$DIR_MLFLOW mlflow experiments create -n $EXPERIMENT_NAME
MLFLOW_TRACKING_URI=$DIR_MLFLOW mlflow run . --experiment-name $EXPERIMENT_NAME --no-conda -P experiment_name=$EXPERIMENT_NAME -P run_name=$RUN_NAME