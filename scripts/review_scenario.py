import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment import digest, write_json
from src.schemas import load_dataset

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--reviewer',required=True)
args=parser.parse_args()
if not args.reviewer.strip():
    parser.error('A reviewer name is required')
dataset=load_dataset(args.directory,args.directory/'claims_metadata.json')
path=args.directory/'scenario.json'
value=json.loads(path.read_text())
value.update(human_reviewed=True,reviewed_dataset_sha256=digest(dataset.model_dump_json()),reviewer=args.reviewer.strip(),reviewed_at=datetime.now(timezone.utc).isoformat())
write_json(path,value)
print('Recorded human review for',value['scenario_id'])
