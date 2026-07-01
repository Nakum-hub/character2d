#!/usr/bin/env python3
"""
Cross-phase stack audit (character2d, Phases 1-8). Verifies the DHPF manifests are internally
consistent and complete: every cross-reference resolves, the firewall holds (physics/performance/
animation write only the Phase-5 surface), single-owner is intact, and the identity foundation
(Phase-1 spec) is present. Emits _reports/audit/stack_audit.md. Reproducible: python3 tools/audit_stack.py
"""
import os, json
CH=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"Character")
def L(n): 
    p=os.path.join(CH,n); return json.load(open(p)) if os.path.exists(p) else None
layers,mesh,rig,params,phys,perf,anim,spec = [L(n) for n in
    ["layers.json","mesh.json","rig.json","params.json","physics.json","performance.json","animation.json","character_spec.json"]]
rows=[]
def check(name, ok, detail): rows.append((name, bool(ok), detail))

# artifacts present (the 8-phase deliverable set)
present={n: L(n) is not None for n in ["character_spec.json","layers.json","mesh.json","rig.json","params.json","physics.json","performance.json","animation.json"]}
check("all_phase_artifacts_present", all(present.values()), ", ".join(f"{k}:{'y' if v else 'MISSING'}" for k,v in present.items()))

defids={d["id"] for d in rig["deformers"]}
pids={p["id"] for p in params["parameters"]}
physin={p["id"] for p in params["parameters"] if p.get("physicsInput")}

# P3 mesh <-> P2 layers
mesh_layer_ids={m["name"] for m in json.load(open(os.path.join(CH,"mesh.json")))["parts"]}
check("mesh_parts_reference_layers", all(any(m["name"]==l["name"] for l in layers) for m in json.load(open(os.path.join(CH,"mesh.json")))["parts"]),
      f"{len(mesh_layer_ids)} meshed parts all trace to layers.json")

# P4 rig params affect existing deformers
bad_affects=[(p["id"],a) for p in rig["parameters"] for a in p.get("affects",[]) if a not in defids]
check("rig_params_affect_existing_deformers", not bad_affects, f"{len(bad_affects)} dangling (sample {bad_affects[:3]})")

# P5 L0 own existing deformers; writes resolve
bad_owns=[(p["id"],p["owns"]) for p in params["parameters"] if p.get("owns") and p["owns"].split(".")[0] not in defids]
dangling=[(p["id"],w["to"]) for p in params["parameters"] for w in p.get("writes",[]) if w["to"] not in pids]
check("L0_owns_existing_deformer", not bad_owns, f"{len(bad_owns)} bad owns")
check("param_writes_resolve", not dangling, f"{len(dangling)} dangling writes")

# single-owner (one deformer channel per L0)
owners={}
dup=[]
for p in params["parameters"]:
    if p.get("owns"):
        if p["owns"] in owners: dup.append(p["owns"])
        owners[p["owns"]]=p["id"]
check("single_owner_channels", not dup, f"{len(owners)} owned channels, {len(dup)} duplicates")

# firewall: physics/performance/animation write only the Phase-5 surface (+ physics->physics-input)
phys_bad=[r["outputParam"] for r in phys["regions"].values() if r["outputParam"] not in physin]
perf_bad=sorted({k for src in [*[e.get("writes",{}) for e in perf["emotionLibrary"].values()],
                               *[c.get("writes",{}) for c in perf["conversationBehaviors"].values()]] for k in src if k not in pids})
anim_bad=sorted({k for g in anim["gestureLibrary"].values() for k in g["curves"] if k not in pids})
check("physics_firewall_physicsinput_only", not phys_bad, f"{len(phys_bad)} non-physics-input outputs")
check("performance_firewall_surface_only", not perf_bad, f"{len(perf_bad)} off-surface writes")
check("animation_firewall_surface_only", not anim_bad, f"{len(anim_bad)} off-surface writes")

# P7 emotions coherence (>=3 body channels) carried into audit
BODY={"P_Body_ChestExpand","P_Body_ChestCompress","P_Body_ShoulderRoll_L","P_Body_ShoulderRoll_R",
      "P_Body_ShoulderElev_L","P_Body_ShoulderElev_R","P_Body_Posture","P_Body_SpineCurve",
      "P_Head_Tilt","P_Head_Forward","P_Head_Back","P_Hand_Form_L","P_Hand_Form_R"}
under=[nm for nm,e in perf["emotionLibrary"].items() if nm!="Neutral" and sum(1 for k in e.get("writes",{}) if k in BODY)<3]
check("emotions_body_coherence", not under, f"{len(under)} emotions with <3 body channels (sample {under[:3]})")

# P8 every animation entry carries the 11-point traceability block
NEED={"intent","emotionalContext","trigger","entryState","exitState","layerParticipation","parameters","physics","blend","failureModes","validation"}
tb=[]
for s,sd in anim["stateMachine"].items(): tb.append(("state:"+s,sd.get("traceability",{})))
for g,gd in anim["gestureLibrary"].items(): tb.append(("gesture:"+g,gd.get("traceability",{})))
missing_tb=[n for n,t in tb if not NEED<=set(t)]
check("animation_11pt_traceability", not missing_tb, f"{len(tb)} entries checked, {len(missing_tb)} incomplete")

# Phase-1 identity: A2 resolved, watch corrected
check("identity_A2_resolved", spec and spec["resolved_decisions"]["A2_height"]["status"]=="RESOLVED",
      spec["resolved_decisions"]["A2_height"]["value"] if spec else "no spec")
check("identity_watch_left", spec and "LEFT" in spec["resolved_decisions"]["watch_side"]["value"], "watch on character-LEFT")

passed=sum(1 for _,ok,_ in rows if ok)
md="# Cross-Phase Stack Audit (Phases 1-8)\n\n"+f"**{passed}/{len(rows)} checks pass.**\n\n| Check | Result | Detail |\n|---|---|---|\n"+\
   "\n".join(f"| {n} | {'✅' if ok else '❌'} | {d} |" for n,ok,d in rows)+"\n"
open(os.path.join(CH,"_reports","audit","stack_audit.md"),"w").write(md)
print(f"STACK AUDIT: {passed}/{len(rows)} pass")
for n,ok,d in rows: print(f"  {'PASS' if ok else 'FAIL'}  {n}: {d}")
