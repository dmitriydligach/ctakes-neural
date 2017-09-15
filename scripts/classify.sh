#!/bin/bash

source $(dirname $0)/env/bin/activate
python $(dirname $0)/ft_classify.py $*
ret=$?
deactivate
exit $ret
