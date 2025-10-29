from utils import *

APPS_ROOT = 'Apps/'
APP_CALLS = {'otp':['java','-Xmx1G','-jar', '"'+APPS_ROOT+'otp-1.2.0-shaded.jar'+'"'],
             'photon':['java','-jar', '"'+APPS_ROOT+'photon-0.3.0'+'"']}



def get_app_call(app):
    # TO DO: make local.
    return APP_CALLS.get(app.lower())

def get_cmd(app, params=[]):
    exe = get_app_call(app)
    if exe:
        cmd = ' '.join(exe+params)
        print(cmd)
        return cmd
    else: print('ERROR: App name "'+app+'" unknown!')    

def run_app(app, params=[], shell=False):
    exe = get_app_call(app)
    if exe:
        call = exe + params
        return subprocess_call(call, shell)
    else: print('ERROR: App name "'+app+'" unknown!')
