"""经本机 qqbot（AstrBot）给绑定账号发一条消息；发送时现读 qqbot 的配置，不另存凭据。"""
import json
import socket
import sys
import urllib.request

QQBOT_DIR = '/home/tcy/Programs/qqbot/monitor'

with open(f'{QQBOT_DIR}/config.json', encoding='utf-8') as f:
    cfg = json.load(f)
with open(f'{QQBOT_DIR}/secrets.json', encoding='utf-8') as f:
    key = json.load(f)['astrbot_api_key']
text = f'【渲染】{socket.gethostname()}\n' + ' '.join(sys.argv[1:])
body = json.dumps({'umo': cfg['target_umo'], 'message': [{'type': 'plain', 'text': text}]}, ensure_ascii=False).encode()
req = urllib.request.Request(cfg['astrbot_url'] + '/im/messages', data=body, method='POST',
                             headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=15) as r:
    if json.load(r).get('status') != 'ok':
        sys.exit('QQ 通知未确认发送成功')
print('QQ 通知已发送')
