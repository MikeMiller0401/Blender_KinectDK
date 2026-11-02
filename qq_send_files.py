"""经本机 qqbot（AstrBot）给绑定账号发文字加图片/视频/文件：qq_send_files.py 说明文字 文件...
发送时现读 qqbot 的配置；上传用文件专用 key，发消息用普通 key，不另存凭据。"""
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

QQBOT_DIR = '/home/tcy/Programs/qqbot/monitor'
KINDS = {'image': {'.png', '.jpg', '.jpeg', '.gif', '.webp'}, 'video': {'.mp4', '.mov', '.webm'}}

with open(f'{QQBOT_DIR}/config.json', encoding='utf-8') as f:
    cfg = json.load(f)
with open(f'{QQBOT_DIR}/secrets.json', encoding='utf-8') as f:
    secrets = json.load(f)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, data, content_type, key):
    req = urllib.request.Request(cfg['astrbot_url'] + path, data=data, method='POST',
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': content_type})
    with opener.open(req, timeout=300) as r:
        res = json.load(r)
    if res.get('status') != 'ok':
        sys.exit(f'{path} 失败: {res}')
    return res.get('data')


parts = [{'type': 'plain', 'text': sys.argv[1]}]
for f in map(Path, sys.argv[2:]):
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{f.name}"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + f.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    att = call('/file', body, f'multipart/form-data; boundary={boundary}', secrets['astrbot_file_api_key'])
    kind = next((k for k, exts in KINDS.items() if f.suffix.lower() in exts), 'file')
    parts.append({'type': kind, 'attachment_id': att['attachment_id']})
call('/im/messages', json.dumps({'umo': cfg['target_umo'], 'message': parts}, ensure_ascii=False).encode(),
     'application/json', secrets['astrbot_api_key'])
print('QQ 已发送', len(parts) - 1, '个附件')
