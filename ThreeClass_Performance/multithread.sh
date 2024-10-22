#!/bin/sh -x

hostname

export TFDIR=$HOME/work/tensorflow

export TFLIB=/gfsvol01/cms/users/dellaric/work/cms/tensorflow

export PYTHONUSERBASE=$TFLIB/local.py3
export PATH=$TFLIB/local.py3/bin:$PATH
export LD_LIBRARY_PATH=$TFLIB/local.py3/lib64

source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh

date

#cd $TFDIR/py3

export TF_NUM_INTRAOP_THREADS=16

export PYTHONUNBUFFERED=1

python3 steer_inputs_DNN_mt.py

date

exit
