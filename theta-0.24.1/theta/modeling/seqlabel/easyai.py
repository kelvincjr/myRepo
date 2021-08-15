# -*- coding: utf-8 -*-
import requests

def easyai_stop_machine(machine_id, machine_key):
    url = 'https://www.easyaiforum.cn/api/control/stop_machine'
    data = {"machine_id": machine_id, "machine_key": machine_key}
    req = requests.post(url, data=data)
    res = req.json()
    if res['code'] == 1000:
        return True
    else:
        return False


def stop_machine():
    machine_id = '3547'
    machine_key = '6e69db81acc5e3dea49c7c1e6f2b87b5'
    easyai_stop_machine(machine_id, machine_key)
