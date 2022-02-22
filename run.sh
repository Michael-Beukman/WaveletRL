export PYTHONPATH=$PYTHONPATH:`pwd`
if [ -f $1 ]; then
    python -u $@
else
    python -u ../$@
fi
