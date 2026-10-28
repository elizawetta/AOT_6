
def set_scores(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else float(not expected)
    recall = overlap / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_record(record: dict, gold: dict, confidence_threshold: float = 0.8):
    answer = record.get('answer') if record.get('valid') else None
    predicted = answer['status'] if answer else None
    precision, recall, f1 = set_scores(answer['evidence_ids'] if answer else [], gold['gold_evidence_ids'])
    counter = set_scores(answer['counterevidence_ids'] if answer else [], gold['gold_counterevidence_ids'])[1]
    if not answer:
        precision = recall = f1 = counter = 0.0
    return dict(status_correct=predicted == gold['gold_status'], evidence_precision=precision, evidence_recall=recall, evidence_f1=f1, counterevidence_recall=counter if gold['gold_counterevidence_ids'] else None, both_correct=predicted == 'both' if gold['gold_status'] == 'both' else None, insufficient_correct=predicted == 'insufficient' if gold['gold_status'] == 'insufficient' else None, confident_when_insufficient=(answer['confidence'] >= confidence_threshold and predicted != 'insufficient') if answer and gold['gold_status'] == 'insufficient' else (False if gold['gold_status'] == 'insufficient' else None), input_tokens=record.get('input_tokens'), output_tokens=record.get('output_tokens'), elapsed_seconds=record.get('elapsed_seconds'), valid=bool(answer))


def summarize(records: list[dict], questions: list[dict]):
    import pandas as pd
    gold = {q['question_id']: q for q in questions}
    rows = []
    for r in records:
        if r['question_id'] not in gold:
            raise ValueError('No gold for result')
        rows.append(dict(mode=r['mode'], question_id=r['question_id'], **evaluate_record(r, gold[r['question_id']])))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame
    summary = frame.groupby('mode', sort=True).agg(status_accuracy=('status_correct','mean'), evidence_precision=('evidence_precision','mean'), evidence_recall=('evidence_recall','mean'), evidence_f1=('evidence_f1','mean'), counterevidence_recall=('counterevidence_recall','mean'), both_recall=('both_correct','mean'), insufficient_recall=('insufficient_correct','mean'), confident_when_insufficient=('confident_when_insufficient','mean'), mean_input_tokens=('input_tokens','mean'), mean_output_tokens=('output_tokens','mean'), mean_seconds=('elapsed_seconds','mean'), valid_response_rate=('valid','mean'), completed_questions=('question_id','count')).reset_index()
    for label in ['both', 'insufficient']:
        values = {}
        for mode in summary['mode']:
            subset = [r for r in records if r['mode'] == mode and r.get('valid') and r['answer']['status'] == label]
            values[mode] = sum(gold[r['question_id']]['gold_status'] == label for r in subset) / len(subset) if subset else float('nan')
        summary[label + '_precision'] = summary['mode'].map(values)
    return frame, summary


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left | right else 1.0


def compare_runs(baseline: list[dict], changed: list[dict]):
    base = {(r['question_id'], r['mode']): r for r in baseline}
    rows = []
    for r in changed:
        key = r['question_id'], r['mode']
        if key not in base:
            continue
        a, b = base[key], r
        valid = a.get('valid', False) and b.get('valid', False)
        aa, bb = a.get('answer') or {}, b.get('answer') or {}
        union = set(aa.get('evidence_ids', [])) | set(bb.get('evidence_ids', []))
        rows.append(dict(question_id=key[0], mode=key[1], comparable=valid, status_changed=(aa.get('status') != bb.get('status')) if valid else None, evidence_jaccard=(len(set(aa.get('evidence_ids', [])) & set(bb.get('evidence_ids', []))) / len(union) if union else 1.0) if valid else None, counterevidence_jaccard=jaccard(aa.get('counterevidence_ids', []), bb.get('counterevidence_ids', [])) if valid else None))
    return rows
