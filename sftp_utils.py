import __future__
import paramiko
# from src.config import log
import glob
import logging 
log = logging.getLogger(__name__)   

local_dir = '/code/code/pyQSM/'
remote_dir = '/home/wischmcj/Desktop/pyQSM/'
host = '192.168.0.157'

def put_file(file, sftp):
    start_loc = f'{local_dir}{file}'
    end_loc = f'{remote_dir}'
    print(f'attempting sftp from {start_loc} to {end_loc}')
    breakpoint()
    try:
        put_result = sftp.put(start_loc, end_loc)
        msg = f'sftp put from {start_loc} to {end_loc}: {put_result}'
    except FileNotFoundError as e:
        msg =f'Error putting to {end_loc} from {start_loc} to: {e}'
    return msg


def get_file(file, sftp):
    start_loc = f'/home/wischmcj/Desktop/pyQSM/{file}'
    end_loc = f'./{file}'
    try:
        get_result = sftp.get(start_loc,end_loc)
    except FileNotFoundError as e:
        print('Error getting file: {e}')
    msg = f'sftp get success from {start_loc} to {end_loc} '
    return msg

def sftp(file,
         wild_card = True,
         action='put'):
    ssh = paramiko.SSHClient() 
    print(f'ssh client to {host} created')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect(host, username='penguaman', password='')
    paramiko.Agent()
    ssh.connect(host, username='',password='')
    print('connected')
    sftp = ssh.open_sftp()
    print('sftp open')
    if wild_card:
        to_send = glob.glob(file)
    else:
        to_send = [file]
    msg='sftp not yet run'
    for file in to_send:
        try:
            if action == 'get':
                print(f'getting files {to_send}')
                msg = get_file(file, sftp)
            else:
                print(f'sending files {to_send}')
                msg = put_file(file, sftp)
        except Exception as e:
            msg=f'Error sftping {file}, {e}, {msg}'
            print(msg)
    print(msg)   

if __name__=='__main__':
    sftp(file='src_pcs/*.py', action='put')