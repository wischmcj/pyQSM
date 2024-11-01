import open3d as o3d

def cube_red():
   print('running cube red')
   cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
   cube_red.compute_vertex_normals()
   cube_red.paint_uniform_color((1.0, 0.0, 0.0))
   o3d.visualization.draw(cube_red)

#o3d.visualization.webrtc_server.enable_webrtc()
def send_ack(data):
    print(f'sending ack {data}')
    cube_red()
    print('returning')
    return "Received WebRTC data channel message with data: " + data

#o3d.visualization.webrtc_server.register_data_channel_message_callback(
#    "webapp/input", send_ack)

if __name__=='__main__':
   print('running main in rtc_server')
   o3d.visualization.webrtc_server.enable_webrtc()
   print('enabled rtc')
   cube_red()
   o3d.visualization.webrtc_server.register_data_channel_message_callback(
    "webapp/input", send_ack)



