#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install -U -r requirements.txt
pip install ipykernel jupyter ipywidgets nbformat
python -m ipykernel install --user --name="kernel"

KALEIDO_FAULTY_SCRIPT_PATH=venv/lib/python3.12/site-packages/kaleido/executable/kaleido
sed -i 's/cd $DIR/cd "$DIR"/g' $KALEIDO_FAULTY_SCRIPT_PATH
sed -i 's/kaleido $@/kaleido "$@"/g' $KALEIDO_FAULTY_SCRIPT_PATH

clear
echo "[SUCCESS] setup completed"