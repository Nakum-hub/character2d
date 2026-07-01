#!/usr/bin/env python3
"""
Full-stack showcase (character2d, Phases 1-8). Drives the assembled DHPF stack (params->rig via
the resolver, Phase-7 emotions, Phase-8 gestures) through the IMPROVED render engine that now
shows body posture (chest openness, lean, head-forward, both-arm raise). Re-verifies Identity-Lock
and renders a montage. Reproducible: python3 tools/showcase.py
"""
import os, json, math, numpy as np, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_engine as R
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw
CH=R.ROOT+"/Character"; REP=CH+"/_reports/audit"; os.makedirs(REP,exist_ok=True)
PARAMS=json.load(open(CH+"/params.json"))["parameters"]; BYID={p["id"]:p for p in PARAMS}; ALLOWED=set(BYID)
PERF=json.load(open(CH+"/performance.json")); EMO=PERF["emotionLibrary"]

def resolve(intent):
    contrib={}; defaults={p["id"]:p["default"] for p in PARAMS if p["tier"]=="L0" and not p.get("physicsInput")}
    def emit(pid,val,vis=()):
        p=BYID.get(pid); 
        if not p or pid in vis: return
        vis=vis+(pid,)
        if p["tier"]=="L0" and not p.get("physicsInput"): contrib.setdefault(pid,[]).append((p["blend"],val))
        for w in p.get("writes",[]): emit(w["to"],val*w.get("gain",1.0),vis)
    res=dict(defaults)
    for pid,val in intent.items(): 
        assert pid in ALLOWED, pid; emit(pid,val)
    for pid,cl in contrib.items():
        p=BYID[pid]; base=p["default"]; add=[v for b,v in cl if b in("additive","signed-sum","distribute")]
        wt=[v for b,v in cl if b=="weighted"]; mx=[v for b,v in cl if b=="max-wins"]; val=base
        if add: val=base+sum(add)
        if wt: val=val+sum(wt)/max(1,len(wt))
        if mx: val=base+max(mx,key=abs)
        lo,hi=p["clampedTo"]; res[pid]=float(max(lo,min(hi,val)))
    return res
def to_p4(L0):
    g=lambda k,d=0:L0.get(k,d)
    return {"ParamAngleX":g("P_Head_RotX"),"ParamAngleY":g("P_Head_RotY"),"ParamAngleZ":g("P_Head_RotZ"),
        "ParamBodyAngleZ":g("P_Body_ChestTwist")/0.40,"ParamBreath":g("P_Body_Breathing"),
        "ParamEyeLookX":g("P_Eye_IrisX_L"),"ParamEyeLookY":g("P_Eye_IrisY_L"),
        "ParamEyeOpenL":g("P_Eye_LidUpper_L",1),"ParamEyeOpenR":g("P_Eye_LidUpper_R",1),
        "ParamMouthForm":g("P_Mouth_LipCorners"),"ParamCheek":max(g("P_Cheek_Raise_L"),g("P_Cheek_Raise_R")),
        "ParamMouthOpenY":g("P_Jaw_Rotation"),"ParamArmRaiseR":g("P_Arm_Raise_R"),
        # body posture (improved engine)
        "ParamChestExpand":g("P_Body_ChestExpand"),"ParamBodyLeanX":g("P_Body_RotX")/10.0,
        "ParamBodyLeanY":g("P_Body_RotY")/10.0,"ParamHeadFwd":g("P_Head_Forward")-g("P_Head_Back"),
        "ParamArmRaiseL":g("P_Arm_Raise_L"),"ParamForearmTwistL":g("P_Arm_ForeTwist_L")}

FRONT=R.FRONT
# Identity-Lock (defaults) still holds with the improved engine
rest=R.render(to_p4(resolve({})),plugs=False); fw=R.over_white(FRONT); rw=R.over_white(rest)
SS=float(ssim(cv2.cvtColor(fw,cv2.COLOR_RGB2GRAY),cv2.cvtColor(rw,cv2.COLOR_RGB2GRAY)))
dmap=np.abs(rw.astype(int)-fw.astype(int)).sum(2); edge=cv2.morphologyEx((FRONT[:,:,3]>40).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((7,7),np.uint8))>0
interior=int(((dmap>30)&~edge).sum())
print(f"IDENTITY-LOCK (improved engine): SSIM {round(SS,4)} interior {interior}px -> {'PASS' if SS>0.995 and interior<500 else 'CHECK'}")

# showcase intents (exercise body-posture channels)
SC={
 "Neutral":{},
 "Confident (chest open)":EMO["Confidence"]["writes"],
 "Pride (chest max)":EMO["Pride"]["writes"],
 "Presenting (open-palm)":{**EMO["Confidence"]["writes"],"P_Arm_Raise_R":0.35,"P_Body_ChestExpand":0.35},
 "Celebrate (both arms)":{"P_Arm_Raise_L":0.9,"P_Arm_Raise_R":0.9,"P_Body_ChestExpand":0.4,"P_Mouth_Smile":0.9,"P_Head_RotX":-4},
 "Lean-in (interest)":{**EMO["Interest"]["writes"],"P_Body_RotY":6,"P_Head_Forward":0.4},
}
frames={}
for nm,intent in SC.items():
    frames[nm]=R.over_white(R.render(to_p4(resolve(intent)),plugs=(nm!="Neutral")))
ims=[Image.fromarray(frames[k]).crop((420,250,1130,1650)) for k in SC]
for im in ims: im.thumbnail((230,430))
cols=3; rows=2; cw=max(i.width for i in ims)+10; chh=max(i.height for i in ims)+22
sh=Image.new("RGB",(cols*cw,rows*chh),(255,255,255)); dr=ImageDraw.Draw(sh)
for i,(im,k) in enumerate(zip(ims,SC)): x=(i%cols)*cw;y=(i//cols)*chh; sh.paste(im,(x+5,y+20)); dr.text((x+5,y+5),k,fill=(0,0,0))
sh.save(REP+"/fullstack_showcase.png")
print("showcase montage saved:", len(SC),"frames")
