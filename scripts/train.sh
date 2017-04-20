#!/bin/bash

source $(dirname $0)/env/bin/activate
python $(dirname $0)/cnn1filter_train.py $*
ret=$?
deactivate
exit $ret
