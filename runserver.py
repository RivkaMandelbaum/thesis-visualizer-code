#!/usr/bin/env python

#-----------------------------------------------------------------------
# runserver.py
# Author: Rivka Mandelbaum
# Based on code written by: Bob Dondero
#-----------------------------------------------------------------------

from sys import argv, exit, stderr, path
from upgraded_visualizer import app
import os

path.append(os.path.dirname('/Desktop/psynet-work/PsyNet/visualizer'))
TEST_DATA_PATH = "./demo-data/"

def main():

    if len(argv) != 3:
        print('Usage: ' + argv[0] + ' port dirname', file=stderr)
        exit(1)

    try:
        port = int(argv[1])
    except Exception:
        print('Port must be an integer.', file=stderr)
        exit(1)

    try:
        if argv[2] in ["-test", "-t"]:
            data_path =  TEST_DATA_PATH
        else:
            data_path = argv[2]
        app.config['data_path'] = data_path
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as ex:
        print(ex, file=stderr)
        exit(1)

if __name__ == '__main__':
    main()
