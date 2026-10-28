import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation import compare_runs, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    if not manifest.get('completed'):
        raise ValueError('Wait for run completion before final analysis')
    records=list(map(json.loads,(args.run/'answers.jsonl').read_text().splitlines()))
    data=json.loads((args.run/'dataset_snapshot.json').read_text())
    per_question,summary=summarize(records,data['questions'])
    per_question.to_csv(args.run/'per_question_metrics.csv',index=False)
    summary.to_csv(args.run/'summary.csv',index=False)
    gold={q['question_id']:q for q in data['questions']}
    valid=[]
    for r in records:
        if r['valid']:
            valid.append(dict(mode=r['mode'],confidence=r['answer']['confidence'],correct=r['answer']['status']==gold[r['question_id']]['gold_status']))
    if valid:
        frame=pd.DataFrame(valid)
        frame['confidence_bin']=pd.cut(frame.confidence,[0,0.25,0.5,0.75,1.0],include_lowest=True)
        calibration=frame.groupby(['mode','confidence_bin'],observed=True).agg(count=('correct','size'),mean_confidence=('confidence','mean'),status_accuracy=('correct','mean')).reset_index()
        calibration.to_csv(args.run/'calibration.csv',index=False)
    review=[]
    for r in records:
        review.append(dict(question_id=r['question_id'],mode=r['mode'],schema_valid=r['valid'],gold_status=gold[r['question_id']]['gold_status'],predicted_status=r['answer']['status'] if r['valid'] else '',human_semantic_review='',hallucinated_fact='',review_notes=''))
    worksheet=args.run/'semantic_review.csv'
    if not worksheet.exists():
        pd.DataFrame(review).to_csv(worksheet,index=False)
    if args.baseline:
        base_manifest=json.loads((args.baseline/'manifest.json').read_text())
        if not base_manifest.get('completed'):
            raise ValueError('Baseline is incomplete')
        for field in ['model_config','system_prompt_sha256']:
            if manifest[field]!=base_manifest[field]:
                raise ValueError(f'Incomparable configurations: {field}')
        baseline=list(map(json.loads,(args.baseline/'answers.jsonl').read_text().splitlines()))
        comparison=pd.DataFrame(compare_runs(baseline,records))
        comparison.to_csv(args.run/'stability.csv',index=False)
        comparison.groupby('mode').agg(comparable_pairs=('comparable','sum'),status_change_rate=('status_changed','mean'),mean_evidence_jaccard=('evidence_jaccard','mean'),mean_counterevidence_jaccard=('counterevidence_jaccard','mean')).to_csv(args.run/'stability_summary.csv')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
