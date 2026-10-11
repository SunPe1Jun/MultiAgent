import argparse,json,shutil
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--pred-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();o=Path(a.out_dir);o.mkdir(parents=True,exist_ok=True)
 for src in Path(a.pred_dir).glob('*.json'):
  d=json.loads(src.read_text()); names=d.get('selected_object_names') or ([d.get('selected_object_name')] if d.get('selected_object_name') else [])
  payload={'row_index':d.get('row_index',0),'event_id':d.get('event_id',''),'response_status':'ok','adjusted_response':{'selected_object_name':names[0] if names else '','selected_object_names':names},'parsed_response':{'selected_object_name':names[0] if names else '','selected_object_names':names}}
  (o/f"row_{int(d.get('row_index',0))}.json").write_text(json.dumps(payload))
if __name__=='__main__':main()
