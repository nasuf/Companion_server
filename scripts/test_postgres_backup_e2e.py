"""Real native backup/restore against an owned, network-isolated synthetic PG.

No prod DSN or application imports. CI runs this against the packaged PG16
image; local PGDATA uses tmpfs to avoid unrelated Docker disk growth.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import postgres_backup as engine


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    name='companion-backup-e2e-'+uuid.uuid4().hex[:8]
    def run(command,input=None):
        return subprocess.run(command,input=input,text=True,capture_output=True,check=True,timeout=180).stdout
    run(['docker','run','-d','--name',name,'--network','none','--memory','512m','--tmpfs','/var/lib/postgresql/data:rw,size=256m','-e','POSTGRES_USER=companion','-e','POSTGRES_DB=companion','-e','POSTGRES_PASSWORD=isolated-only','pgvector/pgvector:0.8.0-pg16'])
    try:
        for _ in range(60):
            # The entrypoint's temporary bootstrap server accepts Unix sockets before
            # database setup completes; TCP opens only on the final postmaster.
            if subprocess.run(['docker','exec',name,'pg_isready','-h','127.0.0.1','-p','5432','-U','companion','-d','companion'],capture_output=True).returncode==0:break
            time.sleep(.5)
        else:raise RuntimeError('isolated source database not ready')
        sql=''.join(f'CREATE TABLE synthetic_{i}(id int primary key);INSERT INTO synthetic_{i} VALUES (42);' for i in range(25))
        run(['docker','exec','-i',name,'psql','-Xq','-v','ON_ERROR_STOP=1','-U','companion','-d','companion'],sql)
        with tempfile.TemporaryDirectory(prefix='companion-backup-e2e-') as tmp:
            root=Path(tmp);env={**os.environ,'POSTGRES_CONTAINER_NAME':name}
            engine.atomic_json(root/'.scheduler.json',{'schema_version':1})
            request=engine.request_backup(root);result=engine.process_request(root,'docker',env)
            backup=result['name'];engine.verify(root,backup,'docker',env)
            run(['docker','exec',name,'createdb','-U','companion','restored'])
            # Binary archive stays private and only crosses the owned container's stdin.
            with (root/backup).open('rb') as source:
                subprocess.run(['docker','exec','-i',name,'pg_restore','-U','companion','-d','restored','--exit-on-error','--no-owner','--no-acl'],stdin=source,capture_output=True,check=True,timeout=180)
            total=run(['docker','exec',name,'psql','-XqAt','-U','companion','-d','restored','-c','SELECT count(*) FROM information_schema.tables WHERE table_schema=\'public\';']).strip()
            value=run(['docker','exec',name,'psql','-XqAt','-U','companion','-d','restored','-c','SELECT id FROM synthetic_24;']).strip()
            assert total=='25' and value=='42'
            assert engine.process_request(root,'docker',env)=={'status':'no_request'}
            proof={'status':'passed','scope':'actual request -> host engine -> PG16 dump/globals/checksum/TOC -> native isolated restore','tables':int(total),'manual_request_consumed_once':True,'restore_status_before_drill':engine.snapshot(root)['backups'][0]['restore']['status']}
            assert proof['restore_status_before_drill']=='not_run'
            (args.output/'backup-e2e.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
    finally:run(['docker','rm','-f','-v',name])

if __name__=='__main__':main()
