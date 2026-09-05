#!/usr/bin/env python3
"""Verify the published twelve-case v41 FMA parent re-audit projection."""

from __future__ import annotations

import argparse, ast, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load_rows(path:Path):
    rows=[]
    for number,line in enumerate(path.read_text().splitlines(),1):
        value=json.loads(line)
        if not isinstance(value,dict): raise ValueError(f"row {number} is not an object")
        rows.append(value)
    return rows

def verify(data:Path)->dict:
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    from open_cake_ir.compiler import Compiler

    summary=json.loads((data/"summary.json").read_text())
    rows=load_rows(data/"results.jsonl")
    clusters=json.loads((ROOT/"docs/data/aka-qualified-ir-v6-review-20260904/ir-gap-clusters.json").read_text())["clusters"]
    expected=set(next(x for x in clusters if x["cluster_id"]=="c050-elementwise-fma-fp32")["case_ids"])
    if len(rows)!=12 or {x["case_id"] for x in rows}!=expected:
        raise ValueError("result IDs do not exactly cover c050")
    counts=Counter(x["terminal_status"] for x in rows)
    if dict(counts)!=summary["counts"]:
        raise ValueError("summary counts differ")
    if summary["compiler_revision_id"]!="open-cake-ir-sm100a-v41":
        raise ValueError("Compiler identity differs")
    compiler=Compiler.load(ROOT,ROOT/"compiler/revision.lock.json")
    scalar_blocked=0
    backend_compile_failed=0
    for row in rows:
        if row["prior_fma_gap_resolved"] is not True or row["gpu_test"]!="not_run":
            raise ValueError(f"claim boundary differs: {row['case_id']}")
        if row["performance_measured"] is not False:
            raise ValueError(f"performance claim differs: {row['case_id']}")
        status=row["terminal_status"]
        if status in {"candidate_lowered_compiled","candidate_semantic_review_rejected"}:
            schedule=json.loads(row["schedule_json"])
            assessment=compiler.assess(schedule)
            if not assessment.accepted or not assessment.lowering_eligible:
                raise ValueError(f"published Schedule no longer lowerable: {row['case_id']}")
            lowered=compiler.lower(assessment);ast.parse(lowered.source)
            fmas=[x for x in schedule["operations"] if x.get("parameters",{}).get("op")=="fma"]
            if not fmas or any(x["parameters"]!={"op":"fma","instruction":{"contract":"ptx.fma.rn.f32"}} for x in fmas):
                raise ValueError(f"FMA spelling differs: {row['case_id']}")
            buffers={x["name"]:x for x in schedule["buffers"]}
            operations={x["id"]:x for x in schedule["operations"]}
            maps={x["operation"]:x for x in schedule["access_maps"]}
            program={x["name"]:x["tile"] for x in (schedule.get("program_map") or {}).get("axes",[])}
            loops={x["name"]:x["tile"] for x in schedule.get("tile_loops",[])}
            shape_violations=[]
            for fma in fmas:
                result_shape=buffers[fma["writes"][0]]["shape"]
                for name in fma["reads"]:
                    if buffers[name]["shape"]!=result_shape:
                        raise ValueError(f"hidden FMA broadcast: {row['case_id']}")
                    producer=next((x for x in operations.values() if name in x["writes"]),None)
                    if producer and producer["kind"]=="load":
                        access=maps[producer["id"]];shape=[]
                        for index in access["indices"]:
                            source=index["source"]
                            if source=="program_tile":shape.append(program[index["name"]])
                            elif source=="loop_tile":shape.append(loops[index["name"]])
                            elif source=="dimension":
                                global_buffer=buffers[access["buffer"]]
                                extent=index.get("extent",global_buffer["shape"][index["dimension"]]-index.get("offset",0))
                                shape.append(extent)
                        if not shape:shape=[1]
                        if shape!=buffers[name]["shape"]:
                            shape_violations.append(name)
            compile_result=row["backend_compile"]
            backend_compile_failed += compile_result["status"]=="failed"
            if status=="candidate_lowered_compiled":
                if shape_violations or compile_result["status"]!="passed":
                    raise ValueError("compiled semantic survivor differs")
            else:
                if not shape_violations or compile_result["status"]!="failed":
                    raise ValueError("semantic rejection evidence differs")
                if row.get("semantic_review",{}).get("status")!="rejected":
                    raise ValueError("semantic rejection receipt missing")
        elif status=="remaining_schedule_ir_gap":
            if row["schedule_json"] is not None or not row["remaining_blockers"]:
                raise ValueError(f"remaining-gap shape differs: {row['case_id']}")
            if any(x["owner"]=="runtime_abi" for x in row["remaining_blockers"]):
                scalar_blocked+=1
        elif status=="authoring_rejected":
            assessment=row["compiler_assessment"]
            if assessment["accepted"] is not False or not assessment["blocking_findings"]:
                raise ValueError(f"authoring rejection differs: {row['case_id']}")
        else:
            raise ValueError(f"unknown terminal status: {status}")
    if scalar_blocked!=5 or summary["runtime_scalar_blocked_count"]!=5:
        raise ValueError("runtime-scalar count differs")
    if counts["candidate_lowered_compiled"]!=4 or summary["semantic_survivor_count"]!=4:
        raise ValueError("survivor count differs")
    if backend_compile_failed!=1 or summary["backend_compile_gap_count"]!=1:
        raise ValueError("backend compile failure count differs")
    if counts["authoring_rejected"]!=2 or summary["authoring_rejected_count"]!=2:
        raise ValueError("authoring rejection count differs")
    if (counts["candidate_semantic_review_rejected"]!=1
            or summary["semantic_review_rejected_count"]!=1):
        raise ValueError("semantic rejection count differs")
    return {"schema":"open-cake.aka-fma-v41-reaudit-verification.v1",
            "status":"accepted","cases":len(rows),"unique_cases":len(expected),
            "counts":dict(counts),"semantic_survivors":4,
            "runtime_scalar_blocked":5,"gpu_test":"not_run",
            "performance_measured":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=verify(a.data.resolve())
    if a.output.exists():raise SystemExit("output exists")
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
