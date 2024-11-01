import __future__
import paramiko
# from src.config import log
import glob
import logging
import os
log = logging.getLogger(__name__)

config_local_dir = '/code/code/pyQSM/'
config_remote_dir = '/home/wischmcj/Desktop/pyQSM/'
config_host = '192.168.0.157'

def put_file(file, sftp, remote_dir, local_dir):
    start_loc = f'{local_dir}{file}'
    end_loc = f'{remote_dir}{file}'
    print(f'attempting sftp from {start_loc} to {end_loc}')
    try:
        put_result = sftp.put(start_loc, end_loc)
        msg = f'sftp put from {start_loc} to {end_loc}: {put_result}'
    except FileNotFoundError as e:
        msg =f'Error putting to {end_loc} from {start_loc} to: {e}'
    return msg


def get_file(file, sftp, remote_dir, local_dir):
    start_loc = f'{remote_dir}{file}'
    end_loc = f'{local_dir}{file}'
    try:
        get_result = sftp.get(start_loc,end_loc)
    except FileNotFoundError as e:
        print('Error getting file: {e}')
    msg = f'sftp get success from {start_loc} to {end_loc} '
    return msg

def sftp(file,
         wildcard = True,
         action='put',
         remote_dir = config_remote_dir, local_dir=config_local_dir,
         host = config_host):
    ssh = paramiko.SSHClient() 
    print(f'ssh client to {host} created')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    paramiko.Agent()
    ssh.connect(host, username=os.environ.get("SFTP_USER"),password=os.environ.get("SFTP_TEMP_PASS"))
    print('connected')
    sftp = ssh.open_sftp()
    print('sftp open')
    if wildcard:
        to_send = glob.glob(file)
    else:
        to_send = [file]
    msg='sftp not yet run'
    print(to_send)
    for file in to_send:
        file = file.replace('./','')
        try:
            if action == 'get':
                print(f'getting files {to_send}')
                msg = get_file(file, sftp,remote_dir,local_dir)
            else:
                print(f'sending files {to_send}')
                msg = put_file(file, sftp,remote_dir,local_dir)
        except Exception as e:
            msg=f'Error sftping {file}, {e}, {msg}'
            print(msg)
    print(msg)   

if __name__=='__main__':
    sftp(file='fragment.ply',
          action='put',
          wildcard=False,
          local_dir='/home/penguaman/open3d_data/download/PLYPointCloud/',)
