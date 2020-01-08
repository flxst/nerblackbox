cd pretrained_models

input=$1

if [ -z "$input" ]; then  # no input
  echo '> get model base'
  wget --no-check-certificate https://storage.googleapis.com/ai-center/2019_06_15/swe-uncased_L-12_H-768_A-12.zip
  unzip swe-uncased_L-12_H-768_A-12.zip

  echo '> get model large'
  wget --no-check-certificate https://storage.googleapis.com/ai-center/2019_06_15/swe-uncased_L-24_H-1024_A-16.zip
  unzip swe-uncased_L-24_H-1024_A-16.zip

elif [ $input == 'base' ]; then
  echo '> get model base'
  wget --no-check-certificate https://storage.googleapis.com/ai-center/2019_06_15/swe-uncased_L-12_H-768_A-12.zip
  unzip swe-uncased_L-12_H-768_A-12.zip

elif [ $input == 'large' ]; then
  echo '> get model large'
  wget --no-check-certificate https://storage.googleapis.com/ai-center/2019_06_15/swe-uncased_L-24_H-1024_A-16.zip
  unzip swe-uncased_L-24_H-1024_A-16.zip

else
  echo '> unknown input argument'

fi

