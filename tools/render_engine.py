"""Shared Phase-4 character render engine (deformer math) used by Phase 5+ validation."""
import os, json, math, numpy as np, cv2
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.metrics import structural_similarity as ssim
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH=os.path.join(ROOT,"Character"); REP=os.path.join(CH,"_reports","rig"); os.makedirs(REP,exist_ok=True)
LAYERS=json.load(open(os.path.join(CH,"layers.json")))
MESH={m["name"]:m for m in json.load(open(os.path.join(CH,"mesh.json")))["parts"]}
FRONT=np.array(Image.open(os.path.join(ROOT,"assets","reference","front.png")).convert("RGBA"))
H,W=FRONT.shape[:2]
byname={l["name"]:l for l in LAYERS}

# ---- reliable canvas pivots (Phase 0/3 landmarks) ----
PIV=dict(neck=(778,642), head=(778,648), chest=(783,950), waist=(783,1255),
         shoulder_R=(600,790), shoulder_L=(965,790), elbow_R=(515,1185), elbow_L=(1045,1185),
         wrist_R=(516,1483), wrist_L=(1044,1483), hip=(780,1630), hip_R=(700,1645), hip_L=(862,1645),
         mouth=(778,566), cheek_R=(706,520), cheek_L=(852,520), eye_R=(716,470), eye_L=(846,470),
         watch=(1052,1420), knee_R=(660,2060), knee_L=(900,2060))


PIV=dict(neck=(778,642), head=(778,648), chest=(783,950), waist=(783,1255),
         shoulder_R=(600,790), shoulder_L=(965,790), elbow_R=(515,1185), elbow_L=(1045,1185),
         wrist_R=(516,1483), wrist_L=(1044,1483), hip=(780,1630), hip_R=(700,1645), hip_L=(862,1645),
         mouth=(778,566), cheek_R=(706,520), cheek_L=(852,520), eye_R=(716,470), eye_L=(846,470),
         watch=(1052,1420), knee_R=(660,2060), knee_L=(900,2060))

def part_geo(l):
    m=MESH.get(l["name"])
    img=np.array(Image.open(os.path.join(ROOT,l["file"])).convert("RGBA"))
    if m and len(m["verts"])>=4:
        V=np.array(m["verts"],float); T=np.array(m["tris"],int)
    else:
        h,w=img.shape[:2]; V=np.array([[0,0],[w,0],[w,h],[0,h]],float); T=np.array([[0,1,2],[0,2,3]])
    off=np.array([l["x"],l["y"]],float)
    return img,V,T,off

def aff(verts,piv,sx=1,sy=1,shear=0,rot=0,tx=0,ty=0):
    c,s=math.cos(rot),math.sin(rot)
    M=np.array([[c,-s],[s,c]])@np.array([[1,shear],[0,1]])@np.array([[sx,0],[0,sy]])
    return (verts-piv)@M.T+piv+np.array([tx,ty])

def deform(name,Vc,P):
    """apply active parameters to canvas verts Vc for a part; returns deformed canvas verts."""
    n=name; V=Vc.copy()
    isHead=any(k in n for k in ["FACE","EYE","BROW","MOUTH","NOSE","HEAD","HAIR","Cheek"]) and "Rear" not in n
    isHairRear="Rear" in n or "ScalpPlug" in n
    isNeck="NECK" in n
    isTorso=any(k in n for k in ["SHIRT","BELT","Deltoid","SLEEVE_Upper","BODY_Hidden","ShoulderSocket","Collar_Interior"])
    isArmR=(n.endswith("_R") and any(k in n for k in ["SLEEVE","FOREARM","WRIST","HAND","Palm","Finger","Thumb","WATCH","Deltoid"]))
    # Body Z twist (distributed by band): horizontal shear/scale about waist
    bz=P.get("ParamBodyAngleZ",0)/30.0
    if bz and (isTorso or isHead or isNeck or isArmR):
        share={"waist":0.30,"chest":0.40,"neck":0.20,"head":0.10}
        # graded by vertical position: lower torso less, head most (cumulative up the cascade)
        ycen=V[:,1].mean()
        frac=np.clip((PIV["hip"][1]-ycen)/(PIV["hip"][1]-PIV["head"][1]),0,1)
        V=aff(V,np.array(PIV["waist"]),sx=1-0.12*abs(bz)*frac,shear=0.10*bz*frac)
    # Head turn X / Y / Z (+ neck 20% share)
    ax=P.get("ParamAngleX",0)/30.0; ay=P.get("ParamAngleY",0)/30.0; az=P.get("ParamAngleZ",0)/20.0
    if isNeck and (ax or az or ay):
        V=aff(V,np.array(PIV["neck"]),rot=math.radians(6*az*0.3),sy=1+0.03*ay*0.2)
    if isHead and (ax or ay or az):
        V=aff(V,np.array(PIV["head"]),sx=1-0.13*abs(ax),shear=0.06*ax,rot=math.radians(7*ax+12*az),
              tx=26*ax, ty=24*ay)
    if isHairRear and ax:
        V=aff(V,np.array(PIV["head"]),tx=30*ax)   # rear hair reveals opposite side
    # Breathing
    br=P.get("ParamBreath",0)
    if br and (isTorso or isHead):
        V=aff(V,np.array(PIV["chest"]),sy=1+0.02*br); 
        if isHead: V[:,1]-=3*br
    # Eye look (iris/pupil/catchlight only)
    if any(k in n for k in ["Iris","Pupil","Catchlight"]):
        V[:,0]+=12*P.get("ParamEyeLookX",0); V[:,1]+=8*P.get("ParamEyeLookY",0)
    # Blink (upper lid + sclera collapse toward lid line)
    side="L" if n.endswith("_L") else "R"
    op=P.get(f"ParamEyeOpen{side}",1)
    if (("Lid" in n) or ("EYE_White" in n) or ("Lashes" in n) or ("Iris" in n) or ("Pupil" in n)) and op<1:
        lid=PIV[f"eye_{side}"][1]-6
        V[:,1]=V[:,1]+(lid-V[:,1])*(1-op)*np.clip((V[:,1]<lid+30),0,1)
    # Smile (mouth form + cheek)
    mf=P.get("ParamMouthForm",0); ck=P.get("ParamCheek",0)
    if "MOUTH" in n and mf:
        cx=PIV["mouth"][0]; V[:,1]-=np.abs(V[:,0]-cx)/40.0*mf  # corners rise
    if "Cheek" in n and (ck or mf>0):
        amt=max(ck,mf*0.6); cp=np.array(PIV[f"cheek_{side}"]); 
        r=np.linalg.norm(V-cp,axis=1,keepdims=True); V=V+(V-cp)*0.10*amt*np.exp(-(r/60))
    # Arm raise R (rotate char-right arm about shoulder; girdle-first ~20deg)
    arR=P.get("ParamArmRaiseR",0)
    if isArmR and arR:
        deg=20+ (130-20)*arR    # girdle 20 then arm
        V=aff(V,np.array(PIV["shoulder_R"]),rot=math.radians(-deg*0.55))
    # Forearm twist R + watch counter-rotate
    ft=P.get("ParamForearmTwistR",0)
    if n.endswith("_R") and ("FOREARM" in n or "WRIST" in n) and ft:
        V=aff(V,np.array(PIV["wrist_R"]),sx=1-0.25*abs(ft))
    if "WATCH" in n and ft:  # counter-rotate keeps face upright -> minimal change
        V=aff(V,np.array(PIV["watch"]),sx=1-0.05*abs(ft))
    # --- body-posture channels (audit add-on; subtle whole-part affines, identity-safe) ---
    isArmL=(n.endswith("_L") and any(k in n for k in ["SLEEVE","FOREARM","WRIST","HAND","Palm","Finger","Thumb","Deltoid"]))
    lx=P.get("ParamBodyLeanX",0); ly=P.get("ParamBodyLeanY",0)
    if (lx or ly):                                   # whole-figure lean about the hip
        V=aff(V,np.array(PIV["hip"]),rot=math.radians(-2.2*ly), ty=-3.0*abs(lx))
    ce=P.get("ParamChestExpand",0)
    if ce and (isTorso or isNeck):                   # chest openness -> widen upper torso
        V=aff(V,np.array(PIV["chest"]),sx=1+0.05*ce)
    hf=P.get("ParamHeadFwd",0)
    if (isHead or isNeck) and hf:                    # head forward(+)/back(-)
        V[:,1]+=8.0*hf
    arL=P.get("ParamArmRaiseL",0)
    if isArmL and arL:                               # left arm raise (mirror of R)
        V=aff(V,np.array(PIV["shoulder_L"]),rot=math.radians((20+(130-20)*arL)*0.55))
    ftL=P.get("ParamForearmTwistL",0)
    if n.endswith("_L") and ("FOREARM" in n or "WRIST" in n) and ftL:
        V=aff(V,np.array(PIV["wrist_L"]),sx=1-0.25*abs(ftL))
    return V

# behind-fill set: only the head scalp plug is shaped well enough to fill a reveal cleanly.
# (neck/socket/body plugs are coarse rectangles in this offline build and peek at edges; the rig
#  still DEFINES them for the runtime, which will use the meshed shapes + shared boundary loops.)
PLUGS=["ScalpPlug"]
def render(P, parts=None, plugs=False):
    canvas=np.zeros((H,W,4),float)
    plug_layers=[l for l in LAYERS if any(k in l["name"] for k in PLUGS) and l["x"]+l["w"]<=W and l["y"]+l["h"]<=H and (l["w"]>4 or l["h"]>4)] if plugs else []
    vis_layers=sorted([l for l in LAYERS if l["VIS"]=="always"],key=lambda l:l["DEPTH"])
    order=plug_layers+vis_layers   # plugs drawn first (behind); fill reveals on motion
    for l in order:
        if parts and l["name"] not in parts: pass
        img,V,T,off=part_geo(l)
        src=V                       # part-local
        dstc=deform(l["name"],V+off,P)   # deformed canvas
        # bbox
        mn=np.floor(dstc.min(0)).astype(int); mx=np.ceil(dstc.max(0)).astype(int)
        mn=np.clip(mn,[0,0],[W-1,H-1]); mx=np.clip(mx,[1,1],[W,H])
        w=mx[0]-mn[0]; h=mx[1]-mn[1]
        if w<2 or h<2: continue
        if np.allclose(dstc, V+off):  # fast path: undeformed
            x,y=int(off[0]),int(off[1]); hh,ww=img.shape[:2]
            x2,y2=min(W,x+ww),min(H,y+hh)
            if x>=0 and y>=0 and x2>x and y2>y:
                sub=img[:y2-y,:x2-x].astype(float)/255.0; a=sub[:,:,3:4]
                canvas[y:y2,x:x2,:3]=sub[:,:,:3]*a+canvas[y:y2,x:x2,:3]*(1-a)
                canvas[y:y2,x:x2,3:4]=a+canvas[y:y2,x:x2,3:4]*(1-a)
            continue
        dl=dstc-mn
        try:
            tf=PiecewiseAffineTransform(); 
            if not tf.estimate(dl,src): raise RuntimeError
            src_im=img.astype(float)/255.0
            src_im[:,:,:3]*=src_im[:,:,3:4]           # premultiply to kill warp halos
            wimg=warp(src_im, tf, output_shape=(h,w),order=1,mode="constant",cval=0)
            aa=wimg[:,:,3:4]; wimg[:,:,:3]=np.divide(wimg[:,:,:3],aa,out=np.zeros_like(wimg[:,:,:3]),where=aa>1e-3)
        except Exception:
            continue
        a=wimg[:,:,3:4]
        canvas[mn[1]:mn[1]+h,mn[0]:mn[0]+w,:3]=wimg[:,:,:3]*a+canvas[mn[1]:mn[1]+h,mn[0]:mn[0]+w,:3]*(1-a)
        canvas[mn[1]:mn[1]+h,mn[0]:mn[0]+w,3:4]=a+canvas[mn[1]:mn[1]+h,mn[0]:mn[0]+w,3:4]*(1-a)
    out=(canvas*255).astype(np.uint8)
    return out

def over_white(r):
    a=r[:,:,3:4].astype(float)/255.0; return (r[:,:,:3]*a+255*(1-a)).astype(np.uint8)

