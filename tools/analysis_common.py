"""Shared helpers for optional offline analysis; no model or registry writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def parameter_payload_digest(path):
    """Burnpack v1 payload for this fixed network layout, excluding random IDs.

    burn-pack 0.22.1 uses a 10-byte header and a 256-byte-aligned data section.
    Whole-file hashes also include per-process parameter IDs in CBOR metadata.
    """
    data = path.read_bytes()
    magic, version, metadata_size = struct.unpack_from("<IHI", data)
    if magic != 0x4255524E or version != 1:
        raise ValueError("unsupported checkpoint format for parameter comparison")
    offset = ((10 + metadata_size + 255) // 256) * 256
    if offset >= len(data):
        raise ValueError("checkpoint has no tensor payload")
    return hashlib.sha256(data[offset:]).hexdigest()

def checkpoint(root,models,identity,iteration):
    direct=Path(models[identity])/'checkpoints'/f'iteration-{iteration:06}'/'model.bpk'
    if direct.exists():return direct
    candidates=sorted((root/'models').glob(identity+'-attempt-*'))
    for path in candidates:
        found=path/'checkpoints'/f'iteration-{iteration:06}'/'model.bpk'
        if found.exists():return found
    raise FileNotFoundError(direct)

def load_scores(report):
    result={};schedule=None
    for evaluation in report['baselines']:
        name=evaluation['summary']['opponent']
        if name in result:raise ValueError('duplicate opponent')
        games=evaluation['games']
        if len(games)!=report['config']['games']:raise ValueError('evaluation config game count mismatch')
        if len(games)==0 or len(games)%2:raise ValueError('analysis needs completed, paired-seat games per opponent')
        keys=[];scores=[]
        for a,b in zip(games[::2],games[1::2]):
            if a['learner_seat']!=0 or b['learner_seat']!=1 or a['config']['game_seed']!=b['config']['game_seed'] or a['config']['agent_seeds']!=list(reversed(b['config']['agent_seeds'])):
                raise ValueError('invalid seat-swapped seed pairing')
            if any(g['score'] not in [0.,0.5,1.]for g in [a,b]):raise ValueError('invalid score')
            keys.append((a['config']['game_seed'],tuple(a['config']['agent_seeds']),a['config']['max_actions'],b['config']['max_actions']))
            scores.append((a['score']+b['score'])/2)
        if schedule is None:schedule=keys
        if keys!=schedule:raise ValueError('opponents have different seed schedules')
        if len(games)!=evaluation['summary']['games']:raise ValueError('game count mismatch')
        if abs(sum(scores)/len(scores)-evaluation['summary']['score'])>1e-12:raise ValueError('score summary mismatch')
        result[name]=(evaluation['baseline'],scores)
    if not result:raise ValueError('empty opponent panel')
    return schedule,result

