#!/bin/bash

python3 -m venv myenv
source myenv/bin/activate
pip install wheel -r scripts/requirements.txt