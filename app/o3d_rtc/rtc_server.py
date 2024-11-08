import open3d as o3d
import logging
log = logging.getLogger(__name__)

def cube_red():
   print('running cube red')
   cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   cube_red.compute_vertex_normals()
   cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   o3d.visualization.draw(cube_red)

#o3d.visualization.webrtc_server.enable_webrtc()
def send_ack(data):
    log.error(f'sending ack {data}')
    cube_red()
    log.error('returning')
    return "Received WebRTC data channel message with data: " + data

#o3d.visualization.webrtc_server.register_data_channel_message_callback(
 #  "webapp/input", send_ack)

if __name__=='__main__':
   o3d.visualization.webrtc_server.enable_webrtc()
  # o3d.visualization.webrtc_server.register_data_channel_message_callback(
  #      "webapp/input", send_ack)
   log.error('set web rtc')
   cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   cube_red.compute_vertex_normals()
   cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   log.error('about to viz')
   o3d.visualization.draw(cube_red)
   #print('running main in rtc_server')
   #print('enabled rtc')
   #o3d.visualization.webrtc_server.register_data_channel_message_callback(
   # "webapp/input", send_ack)
   #from open3d.visualization import webrtc_server
   #webrtc_server.disable_http_handshake()
   #from open3d.web_visualizer import draw
   #cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   #cube_red.compute_vertex_normals()
   #cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   #cube_red()
   #import time  
   #log.error('I sleep')
   #time.sleep(5)
   #o3d.visualization.webrtc_server.enable_webrtc()
   #cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   #cube_red.compute_vertex_normals()
   #cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   #log.error('bout to viz')
   #o3d.visualization.draw(cube_red)
   #cube_blue = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   #cube_blue.compute_vertex_normals()
   #cube_blue.paint_uniform_color((0.0, 0.0, 1.0))
   #draw(cube_blue)


