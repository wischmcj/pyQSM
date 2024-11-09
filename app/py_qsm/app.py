from __future__ import annotations

import os
import io
import sys
import time
import debugpy
# import matplotlib.pyplot as plt
# import numpy as np
from flask import Flask, render_template, current_app

from sftp_utils import sftp
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
import paramiko
import open3d as o3d




sys.path.insert(0, os.path.dirname(os.getcwd()))

app = Flask(__name__)

@app.route('/poly/<model>')
def load_poly(model):
   #see https://github.com/daavoo/aframe-pointcloud-component
    htmlpage = '''
    <!doctype HTML>
    <html>
        <head>
        <title>My Point Cloud Scene</title>
        <script src="https://aframe.io/releases/0.6.0/aframe.min.js"></script>
        <script src="templates/static/js/aframe-pointcloud-component.min.js"></script>
    </head>
    <body>
        <a-scene>
            <a-pointcloud
                scale="0.5 0.5 0.5"
                position="1.5 2 0.5"
                src="url({}.ply)"
                size="0.05"
                depthWrite="false">
            </a-pointcloud>
        </a-scene>
    </body>
    </html>
    '''.format(model)
    return htmlpage


@app.route('/<page>')
def load_page(page):
    return current_app.send_static_file('{}.html'.format(page))

@app.route('/')
def main():
    return render_template('index.html')
    # return current_app.send_static_file('imgs/index.html')

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
   return '<p>Science Bitch!</p>'
   #o3d.visualization.webrtc_server.enable_webrtc()
   #cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   #cube_red.compute_vertex_normals()
   #cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   #o3d.visualization.draw(cube_red)



# removed to allow removal of plotting functions 
# @app.route('/plot/<string:file>')
# def plot_png(file):
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #app.run()

# if __name__ == "__main__":
#     print("Waiting for client to attach...4")
#     min_number = int(input('Please enter the min number: '))
#     max_number = int(input('Please enter the max number: '))
#     if (max_number < min_number):
#             print('Invalid input - shutting down...')
#     else:
#         print('Thanks.'(
#     debugpy.wait_for_client()
