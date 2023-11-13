!/bin/bash
virtualenv -q -p /usr/bin/python3.6 $1
$1/bin/pip install -r /home/defaverifr/24_SIGIR_DFF/requirements.txt
