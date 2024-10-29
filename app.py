from __future__ import annotations

import os
import io
import sys
import time
import debugpy
# from random import randint

# from timeit import timeit
# from time import time 
debugpy.listen(("0.0.0.0", 5678))
# import geopandas as geo
# import matplotlib.pyplot as plt

# import numpy as np
from flask import Flask, render_template
# import pytest


from sftp_utils import sftp
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
import os
import paramiko




sys.path.insert(0, os.path.dirname(os.getcwd()))

app = Flask(__name__)

@app.route('/time/<string:file>')
def time(file):
    string = compare(file)
    return  render_template('index.html', strings=[string])

@app.route('/cases')
def run_cases():
    log.info('Starting something')

    return ''

@app.route('/get/<string:file>')
def sftp_get(file):
    msg = sftp(file, get=True)
    return  msg

@app.route('/put/<string:file>')
def sftp_put(file):
    msg = sftp(file, get=False)
    return  msg



@app.route('/hello')
def hello():
    return 'Hello, World!'


# removed to allow removal of plotting functions 
# @app.route('/plot/<string:file>')
# def plot_png(file):
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')


# @app.route('/test')
# def test():
#     args_str = "test/test_collection_integration.py"
#     args = args_str.split(" ")
#     pytest.main(args)
#     return 'Tested'



if __name__ == '__main__':
    # pre_populate_cache()
    app.run(debug=True, host='0.0.0.0')


# if __name__ == "__main__":
#     print("Waiting for client to attach...4")
#     min_number = int(input('Please enter the min number: '))
#     max_number = int(input('Please enter the max number: '))
#     if (max_number < min_number):
#             print('Invalid input - shutting down...')
#     else:
#         print('Thanks.'(
#     debugpy.wait_for_client()
#     time.sleep(5)
