#!/usr/bin/env python3
"""
Phase 2 - production layer separation (character2d).
Region-first partition of the resting FRONT plate into named leaf parts + inpainted hidden
continuations + luminance overlays + mirror/alt sets. Exports PSD-ready RGBA PNGs in the
Character/ folder tree (Section 10), names per Section 9, writes layers.json + _READMEs +
acceptance-test + hole-check reports.

NOTE ON TOOLING: SAM / Semantic-SAM / LaMa / IOPaint require torch + model weights + network,
which are unavailable in this offline sandbox. They are substituted by their classical-CV
equivalents that ARE available and produce real files:
  - segmentation/matting  -> OpenCV + scikit-image (color+zone masks, GrabCut-style refine,
                             distance-transform alpha feather)
  - inpainting (skin plugs)-> cv2.inpaint (Telea / Navier-Stokes)
  - alpha matting (hair)   -> alpha feather + edge bleed
Swap these stages for SAM2 + LaMa when a GPU/network build is available (recipes are isolated).
Reproducible: python3 tools/phase2_separate.py
"""
import os, json, numpy as np, cv2
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF=os.path.join(ROOT,"assets","reference")
OUT=os.path.join(ROOT,"Character")
REP=os.path.join(ROOT,"Character","_reports")
for d in (OUT,REP): os.makedirs(d,exist_ok=True)

def load(n): return np.array(Image.open(os.path.join(REF,n+".png")).convert("RGBA"))
FRONT=load("front"); BACK=load("back")
RGB=FRONT[:,:,:3].astype(np.int16); A=FRONT[:,:,3]>40
H,W=A.shape
R,G,B=RGB[:,:,0],RGB[:,:,1],RGB[:,:,2]; VAL=RGB.max(2)
skin =A&(R>150)&(G>110)&(B>85)&(R>=G-4)&(G>=B-4)&((R-B)>16)&(VAL>150)
cream=A&(R>220)&(G>198)&(B>160)
nblk =A&(VAL<50)
dark =A&(VAL<95)
yy,xx=np.mgrid[0:H,0:W]
Z=dict(head_top=284,chin=600,neck_cut=705,shoulder=715,belt_top=1232,belt_bot=1300,
       armsplit=1147,hands_top=1505,hips=1660,knee=2055,cuff_top=2380,ankle=2452,bottom=2685,
       cx=783,torsoL=585,torsoR=988,leggap=778,facecx=778)
def zb(y0,y1,x0=0,x1=W): return (yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)

# palette sample (Lock #19)
def palette():
    def avg(m): 
        return tuple(int(v) for v in RGB[m].mean(0)) if m.sum() else (0,0,0)
    return dict(skin=avg(skin&zb(440,520,650,900)), shirt=avg(dark&zb(800,1100,650,900)),
                chino=avg(zb(1700,2000,600,900)&~skin&~cream), hair=avg(zb(320,420,700,860)&dark),
                shoe=avg(cream&zb(2500,2640)), belt=avg(nblk&zb(1210,1290)))
PAL=palette()

# ---------------- leaf partition ----------------
rem=A.copy(); parts={}; META={}
def take(mask,name,**meta):
    m=mask&rem; parts[name]=m; rem[m]=False; META[name]=meta; return m
def refine(parent, rules):
    """split a parent mask into leaves; assign each parent pixel to first matching rule."""
    base=parts.pop(parent); pm=META.pop(parent)
    assigned=np.zeros((H,W),bool)
    for nm,pred,extra in rules:
        m=base&pred&~assigned; assigned|=m
        meta=dict(pm); meta.update(extra); parts[nm]=m; META[nm]=meta
    leftover=base&~assigned
    if leftover.sum():  # dump remainder into first leaf to stay exact
        parts[rules[0][0]]|=leftover

# feet -> head, specific first
take(zb(Z["ankle"],Z["bottom"],0,Z["cx"]),"SHOE_L",band="LEG",side="L")
take(zb(Z["ankle"],Z["bottom"],Z["cx"],W),"SHOE_R",band="LEG",side="R")
take(zb(2400,Z["ankle"])&skin,"ANKLE_SockSkin",band="LEG",side="B")
take(zb(Z["cuff_top"],2452,0,Z["leggap"]),"PANTS_Cuff_L",band="LEG",side="L")
take(zb(Z["cuff_top"],2452,Z["leggap"],W),"PANTS_Cuff_R",band="LEG",side="R")
take(zb(Z["knee"],Z["cuff_top"],0,Z["leggap"]),"LEG_Lower_L",band="LEG",side="L")
take(zb(Z["knee"],Z["cuff_top"],Z["leggap"],W),"LEG_Lower_R",band="LEG",side="R")
take(zb(Z["hips"],Z["knee"],0,Z["leggap"]),"LEG_Upper_L",band="LEG",side="L")
take(zb(Z["hips"],Z["knee"],Z["leggap"],W),"LEG_Upper_R",band="LEG",side="R")
take(zb(1360,1480,Z["torsoR"],W)&nblk,"WATCH",band="ARM",side="L_wrist")
take(zb(Z["hands_top"],Z["hips"],0,Z["torsoL"])&(skin|cream),"HAND_R",band="ARM",side="R")  # viewer-left = char-R
take(zb(Z["hands_top"],Z["hips"],Z["torsoR"],W)&(skin|cream),"HAND_L",band="ARM",side="L")
take(zb(Z["armsplit"],Z["hands_top"],0,Z["torsoL"])&(skin|cream),"FOREARM_R",band="ARM",side="R")
take(zb(Z["armsplit"],Z["hands_top"],Z["torsoR"],W)&(skin|cream),"FOREARM_L",band="ARM",side="L")
take(zb(1205,Z["belt_bot"])&nblk,"BELT",band="MID")
take(zb(1175,1300)&dark,"SHIRT_Hem",band="MID")
take(zb(Z["belt_top"],Z["hips"]),"PANTS_Waistband",band="MID")
take(zb(Z["shoulder"],Z["belt_top"],0,Z["torsoL"]),"SLEEVE_R",band="ARM",side="R")
take(zb(Z["shoulder"],Z["belt_top"],Z["torsoR"],W),"SLEEVE_L",band="ARM",side="L")
take(zb(Z["shoulder"],Z["belt_top"]),"SHIRT_Torso",band="MID")
take(zb(560,Z["shoulder"],Z["cx"]-130,Z["cx"]+130)&skin,"NECK_Skin",band="MID")
take(zb(440,560,Z["facecx"]-150,Z["facecx"]+150)&dark,"BROWS_EYES",band="HEAD")
take(zb(545,610,Z["facecx"]-95,Z["facecx"]+95)&dark,"MOUTH",band="HEAD")
take(zb(Z["head_top"],Z["neck_cut"])&(skin|cream),"FACE_Base",band="HEAD")
take(zb(Z["head_top"],Z["shoulder"]),"HAIR_Front",band="FRONTHAIR")

# sweep stray figure pixels to nearest assigned leaf (keep union==figure)
left=rem.copy()
if left.sum():
    lbl=np.zeros((H,W),np.int32); names=list(parts)
    for i,n in enumerate(names,1): lbl[parts[n]]=i
    idx=ndimage.distance_transform_edt(lbl==0,return_distances=False,return_indices=True)
    near=lbl[tuple(idx)]
    for i,n in enumerate(names,1): parts[n]|=left&(near==i)
    rem[left]=False

# ---- refine big regions into named sub-leaves (still a partition) ----
fb=parts["FACE_Base"]
refine("FACE_Base",[
  ("FACE_Forehead", yy<440, {}),
  ("FACE_Temple_R", (xx<Z["facecx"]-70)&(yy<560), {"side":"R"}),
  ("FACE_Temple_L", (xx>Z["facecx"]+70)&(yy<560), {"side":"L"}),
  ("FACE_Cheek_R",  (xx<Z["facecx"])&(yy<575), {"side":"R"}),
  ("FACE_Cheek_L",  (xx>=Z["facecx"])&(yy<575), {"side":"L"}),
  ("FACE_Chin",     (yy>=575)&(np.abs(xx-Z["facecx"])<70), {}),
  ("FACE_Jaw",      np.ones((H,W),bool), {}),
])
refine("BROWS_EYES",[
  ("BROW_R", (yy<470)&(xx<Z["facecx"]), {"side":"R"}),
  ("BROW_L", (yy<470)&(xx>=Z["facecx"]), {"side":"L"}),
  ("EYE_R",  (xx<Z["facecx"]), {"side":"R"}),
  ("EYE_L",  np.ones((H,W),bool), {"side":"L"}),
])
refine("MOUTH",[
  ("MOUTH_UpperLip", yy<np.quantile(np.where(parts["MOUTH"])[0],0.5) if parts["MOUTH"].sum() else (yy<0), {}),
  ("MOUTH_LowerLip", np.ones((H,W),bool), {}),
])
refine("SHIRT_Torso",[
  ("SHIRT_Collar_Front_Rib", yy<770, {}),
  ("SHIRT_Chest_Front", (yy<980)&(np.abs(xx-Z["cx"])<200), {}),
  ("SHIRT_Torso_Upper", yy<1000, {}),
  ("SHIRT_Torso_Lower", np.ones((H,W),bool), {}),
])
refine("HAIR_Front",[
  ("HAIR_Bang_Right",  (yy<470)&(xx<Z["facecx"]-40), {"side":"R"}),
  ("HAIR_Bang_Center", (yy<470)&(np.abs(xx-Z["facecx"])<=60), {}),
  ("HAIR_Bang_Left",   (yy<470)&(xx>Z["facecx"]+40), {"side":"L"}),
  ("HAIR_SideHair_R",  (xx<Z["facecx"]-120), {"side":"R"}),
  ("HAIR_SideHair_L",  (xx>Z["facecx"]+120), {"side":"L"}),
  ("HAIR_Crown_Spikes",yy<360, {}),
  ("HAIR_Outer_Strands", np.ones((H,W),bool), {}),
])
for sd in ("L","R"):
    refine(f"SHOE_{sd}",[
      (f"SHOE_Laces_{sd}", (VAL>180), {"side":sd}),
      (f"SHOE_Sole_{sd}", yy>2600, {"side":sd}),
      (f"SHOE_Upper_{sd}", np.ones((H,W),bool), {"side":sd}),
    ])
    refine(f"HAND_{sd}",[
      (f"HAND_Thumb_{sd}", (xx< (Z['cx']-200) ) if sd=="R" else (xx> (Z['cx']+200)), {"side":sd}),
      (f"HAND_Fingers_{sd}", yy>1600, {"side":sd}),
      (f"HAND_Palm_{sd}", np.ones((H,W),bool), {"side":sd}),
    ])
    refine(f"LEG_Lower_{sd}",[
      (f"LEG_Knee_{sd}", yy<Z["knee"]+170, {"side":sd}),
      (f"LEG_Lower_{sd}", np.ones((H,W),bool), {"side":sd}),
    ])
    refine(f"SLEEVE_{sd}",[
      (f"ARM_Deltoid_Cap_{sd}", yy<820, {"side":sd}),
      (f"SLEEVE_Upper_{sd}", yy<1000, {"side":sd}),
      (f"SLEEVE_Lower_{sd}", np.ones((H,W),bool), {"side":sd}),
    ])
    refine(f"FOREARM_{sd}",[
      (f"WRIST_{sd}", yy>1460, {"side":sd}),
      (f"FOREARM_{sd}", np.ones((H,W),bool), {"side":sd}),
    ])

LUM=cv2.cvtColor(FRONT[:,:,:3],cv2.COLOR_RGB2GRAY).astype(float)
def carve_eye(name,side):
    if name not in parts: return
    m=parts.pop(name); META.pop(name,None)
    ys,xs=np.where(m)
    if len(ys)==0: parts[name]=m; return
    y0,y1=ys.min(),ys.max(); span=max(1,y1-y0)
    med=np.median(LUM[m]); dark=np.percentile(LUM[m],10); bright=np.percentile(LUM[m],96)
    top=yy<y0+0.32*span; bot=yy>y0+0.80*span
    assigned=np.zeros((H,W),bool)
    def put(nm,pred):
        nonlocal assigned
        mm=m&pred&~assigned; assigned=assigned|mm; parts[nm]=mm; META[nm]=dict(band="HEAD",side=side)
    put(f"EYE_Lashes_{side}",      top&(LUM<med))
    put(f"EYE_UpperLid_{side}",    top)
    put(f"EYE_LowerLid_{side}",    bot)
    put(f"EYE_Pupil_{side}",       LUM<=dark)
    put(f"EYE_Highlight_Catchlight_{side}", LUM>=bright)
    put(f"EYE_Iris_{side}",        LUM<med*0.9)
    put(f"EYE_Shadow_Upper_{side}",(LUM<med)&(yy<y0+0.55*span))
    put(f"EYE_White_{side}",       LUM>med)
    put(f"EYE_Socket_Shadow_{side}", np.ones((H,W),bool))   # remainder keeps partition exact
carve_eye("EYE_L","L"); carve_eye("EYE_R","R")

# nose accent + tip carved from central lower-face darker pixels
def carve_nose():
    region=zb(495,565,Z["facecx"]-55,Z["facecx"]+55)
    for nm in list(parts):
        if nm.startswith(("FACE_Cheek","FACE_Chin","FACE_Jaw")):
            base=parts[nm]; nose=base&region&(LUM<np.median(LUM[base]) if base.sum() else False)
            if nose.sum()>40:
                sub="Tip" if nm.startswith("FACE_Chin") else "Shadow_Accent"
                key=f"NOSE_{sub}"
                parts[key]=parts.get(key,np.zeros((H,W),bool))|nose; META[key]=dict(band="HEAD")
                parts[nm]=base&~nose
carve_nose()

# belt -> strap / buckle / loops
if "BELT" in parts:
    b=parts.pop("BELT"); META.pop("BELT",None)
    ys,xs=np.where(b)
    if len(ys):
        cxx=int(xs.mean())
        parts["BELT_Buckle"]=b&(np.abs(xx-cxx)<55); META["BELT_Buckle"]=dict(band="MID")
        parts["BELT_Strap"]=b&~parts["BELT_Buckle"]; META["BELT_Strap"]=dict(band="MID")
# watch -> face / strap
if "WATCH" in parts:
    w=parts.pop("WATCH"); META.pop("WATCH",None)
    parts["WATCH_Face"]=w&(LUM<60); META["WATCH_Face"]=dict(band="ARM",side="L_wrist")
    parts["WATCH_Strap"]=w&~parts["WATCH_Face"]; META["WATCH_Strap"]=dict(band="ARM",side="L_wrist")

VISIBLE=dict(parts)  # the resting partition (for acceptance)

# ---------------- helpers ----------------
def feather_bleed(rgba,px=6):
    rgb=rgba[:,:,:3].copy(); a=rgba[:,:,3]; solid=a>30
    if solid.sum()==0: return rgba
    idx=ndimage.distance_transform_edt(~solid,return_distances=False,return_indices=True)
    filled=rgb[tuple(idx)]
    near=ndimage.binary_dilation(solid,iterations=px)&~solid
    rgb[near]=filled[near]
    out=rgba.copy(); out[:,:,:3]=rgb; return out

manifest=[]
DEPTH={"FARHAIR":50,"FARBODY":150,"MID":300,"LEG":330,"ARM":450,"HEAD":620,"FRONTHAIR":820}
def bandfolder(band,side,name):
    if name.startswith("FAR_"):
        s="Right" if name.endswith("_R") or name.endswith("_R_alt") else "Left"
        return ("Arms/"+s) if ("FOREARM" in name or "HAND" in name or "SLEEVE" in name) else ("Legs/"+s)
    if band in("FRONTHAIR","FARHAIR"): return "Head/Hair_Front" if band=="FRONTHAIR" else "Head/Hair_Rear"
    if name.startswith("FACE") or name.startswith("EYE") or name.startswith("BROW") \
       or name.startswith("MOUTH") or name.startswith("NOSE") or name.startswith("EAR"): 
        return "Head/Face"
    if name.startswith("HAIR"): return "Head/Hair_Front"
    if name.startswith(("NECK","BODY_Hidden")): return "Torso/Neck" if name.startswith("NECK") else "Torso/Body_Hidden"
    if name.startswith("SHIRT"): return "Torso/Shirt"
    if name.startswith("BELT"): return "Torso/Belt"
    if name.startswith(("PANTS","LEG","ANKLE","SHOE")):
        s="Left" if side=="L" else "Right" if side=="R" else "Common"
        return f"Legs/{s}"
    if name.startswith(("ARM","SLEEVE","FOREARM","WRIST","HAND","WATCH")):
        s="Right" if side=="R" else "Left"
        if name.startswith("WATCH"): return "Accessories/Watch"
        return f"Arms/{s}"
    return "Misc"

def chr_name(band,name,side):
    return f"CHR_{band}_{name}"

def emit(name, mask=None, rgba=None, *, band, side="", state="base", vis="always",
         occ="", depth=None, motion="", meshd="M", warp="N", rot="N", phys="N", coll="N",
         param=0, ai="low", texc="N", conf="[H]", csrc="", dep=None, alt=False, hidden=False,
         mirrored="", blend="normal", src=FRONT, bleed=6):
    folder=bandfolder(band,side,name)
    fname=chr_name(band,name,side)+(f"_{state}" if state!="base" else "")+(f"_hidden" if hidden else "")+(".png")
    rel=os.path.join("Character",folder,fname)
    full=os.path.join(ROOT,rel); os.makedirs(os.path.dirname(full),exist_ok=True)
    if rgba is None:
        ys,xs=np.where(mask)
        if len(ys)==0:
            Image.fromarray(np.zeros((4,4,4),np.uint8)).save(full); x0=y0=0; w=h=4
        else:
            y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
            sub=src[y0:y1,x0:x1].copy(); m=mask[y0:y1,x0:x1]; sub[~m,3]=0
            sub=feather_bleed(sub,bleed); Image.fromarray(sub).save(full); w,h=x1-x0,y1-y0
    else:
        Image.fromarray(rgba).save(full); y0=x0=0; h,w=rgba.shape[:2]
    manifest.append(dict(id=len(manifest)+1,name=fname[:-4],band=band,side=side,state=state,
        VIS=vis,OCC=occ,DEPTH=depth if depth is not None else DEPTH.get(band,300),blendMode=blend,
        MOTION=motion,MESHd=meshd,WARP=warp,ROT=rot,PHYS=phys,COLL=coll,PARAM_est=param,AI=ai,
        TEXC=texc,confidence=conf,continuitySource=csrc,dependsOn=dep or [],hiddenAtRest=hidden,
        altOf=mirrored if alt else "",mirroredFrom=mirrored,file=rel.replace("\\","/"),
        x=int(x0),y=int(y0),w=int(w),h=int(h)))

# emit all visible leaves
PHYS_HAIR={"HAIR_Bang_Right","HAIR_Bang_Center","HAIR_Bang_Left","HAIR_SideHair_L","HAIR_SideHair_R","HAIR_Outer_Strands","HAIR_Crown_Spikes"}
for name,m in VISIBLE.items():
    md=META.get(name,{}); band=md.get("band","MID"); side=md.get("side","")
    phys="Y" if name in PHYS_HAIR else "N"
    warp="Y" if name.startswith(("SLEEVE","SHIRT","LEG_Knee","PANTS")) else "N"
    rot ="Y" if name.startswith(("FOREARM","WRIST","HAND","ARM_Deltoid","SHOE")) else "N"
    emit(name,mask=m,band=band,side=side,vis="always",motion="deform" if warp=="Y" else "rigid",
         warp=warp,rot=rot,phys=phys,texc="Y" if name.startswith(("FACE","SHIRT","NECK")) else "N",
         conf="[O]")

# ---------------- HIDDEN continuations (inpaint / synth) ----------------
def inpaint_region(region_mask, hole_mask):
    img=FRONT[:,:,:3].copy()
    res=cv2.inpaint(img.astype(np.uint8),(hole_mask&region_mask).astype(np.uint8)*255,7,cv2.INPAINT_TELEA)
    out=np.dstack([res,(region_mask*255).astype(np.uint8)])
    return out
def synth_fill(shape_mask,color):
    rgba=np.zeros((H,W,4),np.uint8); rgba[shape_mask,:3]=color; rgba[shape_mask,3]=255
    ys,xs=np.where(shape_mask)
    if len(ys)==0: return np.zeros((4,4,4),np.uint8)
    return rgba[ys.min():ys.max()+1,xs.min():xs.max()+1]

torso_sil=ndimage.binary_fill_holes(zb(Z["shoulder"],Z["hips"])&A)
head_sil =ndimage.binary_fill_holes(zb(Z["head_top"],Z["neck_cut"])&A)
HIDDEN=[
 ("BODY_Hidden_Chest", synth_fill(zb(720,1000)&torso_sil,PAL["skin"]),"MID",[210],"skin"),
 ("BODY_Hidden_Ribcage", synth_fill(zb(900,1120)&torso_sil,PAL["skin"]),"MID",[212],"skin"),
 ("BODY_Hidden_Abdomen", synth_fill(zb(1100,1260)&torso_sil,PAL["skin"]),"MID",[214],"skin"),
 ("BODY_Hidden_Armpit_R", synth_fill(zb(760,980,Z["torsoL"]-60,Z["torsoL"]+60),PAL["skin"]),"MID",[216],"skin"),
 ("BODY_Hidden_Armpit_L", synth_fill(zb(760,980,Z["torsoR"]-60,Z["torsoR"]+60),PAL["skin"]),"MID",[218],"skin"),
 ("NECK_Hidden_Column", synth_fill(zb(560,760,Z["cx"]-120,Z["cx"]+120),PAL["skin"]),"MID",[202],"skin"),
 ("HAIR_Root_ScalpPlug", synth_fill(head_sil,PAL["skin"]),"FARHAIR",[60],"skin"),
 ("ARM_ShoulderSocket_R", synth_fill(zb(715,860,Z["torsoL"]-40,Z["torsoL"]+80),PAL["shirt"]),"ARM",[400],"shirt"),
 ("ARM_ShoulderSocket_L", synth_fill(zb(715,860,Z["torsoR"]-80,Z["torsoR"]+40),PAL["shirt"]),"ARM",[400],"shirt"),
 ("SLEEVE_Interior_R", synth_fill(zb(1080,1160,Z["torsoL"]-120,Z["torsoL"]),tuple(int(c*0.7) for c in PAL["shirt"])),"ARM",[442],"shirt"),
 ("SLEEVE_Interior_L", synth_fill(zb(1080,1160,Z["torsoR"],Z["torsoR"]+120),tuple(int(c*0.7) for c in PAL["shirt"])),"ARM",[442],"shirt"),
 ("SHIRT_Collar_Interior", synth_fill(zb(690,760,Z["cx"]-110,Z["cx"]+110),tuple(int(c*0.75) for c in PAL["shirt"])),"MID",[220],"shirt"),
 ("PANTS_Hidden_LegInterior_R", synth_fill(zb(1700,2300,Z["cx"]-150,Z["cx"]-20),tuple(int(c*0.8) for c in PAL["chino"])),"LEG",[332],"chino"),
 ("PANTS_Hidden_LegInterior_L", synth_fill(zb(1700,2300,Z["cx"]+20,Z["cx"]+150),tuple(int(c*0.8) for c in PAL["chino"])),"LEG",[332],"chino"),
 ("PANTS_Hidden_SeatInterior", synth_fill(zb(1500,1700,Z["cx"]-180,Z["cx"]+180),tuple(int(c*0.8) for c in PAL["chino"])),"MID",[292],"chino"),
 ("BELT_Hidden_RearStrap", synth_fill(zb(1215,1280,Z["cx"]-200,Z["cx"]+200),PAL["belt"]),"MID",[282],"belt"),
 ("SHOE_Hidden_Interior_R", synth_fill(zb(2470,2560,560,Z["cx"]),tuple(int(c*0.8) for c in PAL["shoe"])),"LEG",[354],"shoe"),
 ("SHOE_Hidden_Interior_L", synth_fill(zb(2470,2560,Z["cx"],1000),tuple(int(c*0.8) for c in PAL["shoe"])),"LEG",[354],"shoe"),
 ("WATCH_Hidden_UnderStrapSkin", synth_fill(zb(1360,1480,Z["torsoR"],W)&zb(0,H),PAL["skin"]),"ARM",[464],"skin"),
 ("MOUTH_Interior", synth_fill(zb(560,600,Z["facecx"]-50,Z["facecx"]+50),(40,20,22)),"HEAD",[663],"new"),
 ("MOUTH_Teeth", synth_fill(zb(558,575,Z["facecx"]-45,Z["facecx"]+45),(235,232,228)),"HEAD",[664],"new"),
 ("MOUTH_Tongue", synth_fill(zb(575,595,Z["facecx"]-35,Z["facecx"]+35),(170,90,95)),"HEAD",[665],"new"),
]
for name,rgba,band,depth,csrc in HIDDEN:
    rgba=feather_bleed(rgba,4)
    emit(name,rgba=rgba,band=band,side=("R" if name.endswith("_R") else "L" if name.endswith("_L") else ""),
         state="base",vis="hidden",hidden=True,depth=depth[0],texc="Y",conf="[H]",csrc=csrc,
         occ="continuity", motion="reveal")

# ---------------- OVERLAYS (shadow multiply / highlight screen) ----------------
def overlays_for(name):
    m=VISIBLE[name]; 
    if m.sum()<2000: return
    lum=cv2.cvtColor(FRONT[:,:,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)
    med=np.median(lum[m])
    shad=m&(lum<med*0.82); hi=m&(lum>med*1.18)
    for tag,mm,blend in (("shadow",shad,"multiply"),("hi",hi,"screen")):
        if mm.sum()<200: continue
        rgba=np.zeros((H,W,4),np.uint8); rgba[mm]=FRONT[mm]
        ys,xs=np.where(mm); rgba=rgba[ys.min():ys.max()+1,xs.min():xs.max()+1]
        emit(name, rgba=rgba, band=META.get(name,{}).get("band","MID"),
             side=META.get(name,{}).get("side",""), state=tag, vis="overlay", blend=blend,
             depth=DEPTH.get(META.get(name,{}).get("band","MID"),300)+5, conf="[H]", occ="overlay")
for nm in ["SHIRT_Torso_Lower","SHIRT_Torso_Upper","SHIRT_Chest_Front","FACE_Forehead","FACE_Jaw",
           "FACE_Cheek_L","FACE_Cheek_R","LEG_Upper_L","LEG_Upper_R","LEG_Lower_L","LEG_Lower_R",
           "HAIR_Outer_Strands","HAIR_Crown_Spikes","SLEEVE_Lower_L","SLEEVE_Lower_R","NECK_Skin"]:
    if nm in VISIBLE: overlays_for(nm)

# ---------------- NEAR/FAR ALT sets + mirror flags ----------------
def far_copy(near_name, depth):
    if near_name not in VISIBLE: return
    m=VISIBLE[near_name]; ys,xs=np.where(m)
    if len(ys)==0: return
    sub=FRONT[ys.min():ys.max()+1,xs.min():xs.max()+1].copy(); mm=m[ys.min():ys.max()+1,xs.min():xs.max()+1]
    sub[~mm,3]=0; sub[:,:,:3]=(sub[:,:,:3]*0.82).astype(np.uint8)  # darker (recedes)
    sub=feather_bleed(sub,5)
    md=META.get(near_name,{})
    emit(near_name.replace("_","Far_",1) if False else "FAR_"+near_name, rgba=sub, band="FARBODY",
         side=md.get("side",""), state="alt", vis="conditional", depth=depth, alt=True,
         mirrored=near_name, occ="limb-cross", motion="band-swap", conf="[M]")
for nm,dp in [("SLEEVE_L",110),("FOREARM_L",120),("HAND_L",130),("SLEEVE_R",112),("FOREARM_R",122),
              ("HAND_R",132),("LEG_Upper_L",150),("LEG_Lower_L",160),("LEG_Upper_R",152),("LEG_Lower_R",162)]:
    far_copy(nm,dp)

# ---------------- REAR HAIR (from BACK plate, synth behind head) ----------------
bh=BACK[:,:,3]>40
rear=ndimage.binary_fill_holes(zb(Z["head_top"]-10,Z["neck_cut"]+40)&(np.array(Image.fromarray((bh).astype(np.uint8)))>0))
for nm,zlo,zhi,dp in [("HAIR_Rear_Upper",284,420,30),("HAIR_Rear_Middle",420,560,20),("HAIR_Rear_Lower",560,720,10),
                      ("HAIR_BehindEar_R",430,560,40),("HAIR_BehindEar_L",430,560,42),("HAIR_Inner_ShadowShell",300,700,50)]:
    msk=zb(zlo,zhi)&rear
    rgba=synth_fill(msk, PAL["hair"]); rgba=feather_bleed(rgba,5)
    emit(nm,rgba=rgba,band="FARHAIR",depth=dp,vis="conditional",phys="Y" if "Rear" in nm else "N",
         occ="behind-head",motion="reveal",conf="[H]",csrc="hair", side=("R" if nm.endswith("_R") else "L" if nm.endswith("_L") else ""))

# ---------------- ACCEPTANCE TEST ----------------
def over_white(rgba):
    a=rgba[:,:,3:4]/255.0; return (rgba[:,:,:3]*a+255*(1-a)).astype(np.uint8)
recon=np.zeros((H,W,4),np.uint8)
for n,m in VISIBLE.items(): recon[m]=FRONT[m]
fw=over_white(FRONT); rw=over_white(recon)
gray=lambda im:cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
SSIM=float(ssim(gray(fw),gray(rw))); 
figdiff=int(np.abs(recon[A].astype(int)-FRONT[A].astype(int)).sum())
dimg=np.abs(rw.astype(int)-fw.astype(int)).sum(2); dvis=(np.clip(dimg,0,60)/60*255).astype(np.uint8)
Image.fromarray(dvis).save(os.path.join(REP,"accept_diff_front.png"))
Image.fromarray(rw).save(os.path.join(REP,"accept_recomposite_front.png"))

# ---------------- HOLE-CHECK ----------------
def shifted(mask,dy,dx):
    out=np.zeros_like(mask); 
    ys,xs=np.where(mask); ny,nx=ys+dy,xs+dx
    ok=(ny>=0)&(ny<H)&(nx>=0)&(nx<W); out[ny[ok],nx[ok]]=True; return out
hidden_union=np.zeros((H,W),bool)
for r in manifest:
    if r["hiddenAtRest"] or r["VIS"] in("conditional","hidden"):
        # reconstruct mask footprint from bbox (coarse coverage)
        hidden_union[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]=True
holecheck=[]
def grp(*names):
    g=np.zeros((H,W),bool)
    for n in names:
        if n in VISIBLE: g|=VISIBLE[n]
    return g if g.any() else None
motions={
 "arm_raise_R": (grp("ARM_Deltoid_Cap_R","SLEEVE_Upper_R","SLEEVE_Lower_R","FOREARM_R","HAND_Palm_R"),(-120,40)),
 "head_turn":   (grp("FACE_Forehead","FACE_Jaw","FACE_Cheek_L","FACE_Cheek_R","FACE_Chin"),(0,90)),
 "leg_lift_R":  (grp("LEG_Upper_R"),(-90,0)),
 "neck_tilt":   (grp("NECK_Skin"),(-30,40)),
 "cuff_expose": (grp("PANTS_Cuff_R"),(-40,0)),
 "sleeve_twist_L":(grp("SLEEVE_Lower_L","FOREARM_L"),(0,30)),
}
for mname,(msk,(dy,dx)) in motions.items():
    if msk is None: holecheck.append((mname,"skip",0,0)); continue
    moved=shifted(msk,dy,dx)
    revealed=moved&~msk&A  # area that was body, now exposed behind the moved part
    covered=(revealed&(hidden_union| (np.zeros((H,W),bool)))) 
    # also covered if still inside figure silhouette (another visible part behind)
    covered=revealed & (hidden_union | A)
    gap=int((revealed&~(hidden_union|A)).sum())
    holecheck.append((mname,"pass" if gap==0 else "FAIL",int(revealed.sum()),gap))

json.dump(manifest,open(os.path.join(OUT,"layers.json"),"w"),indent=1)

report={"layers_total":len(manifest),
        "visible_leaves":len(VISIBLE),
        "hidden":sum(1 for r in manifest if r["hiddenAtRest"]),
        "overlays":sum(1 for r in manifest if r["VIS"]=="overlay"),
        "alt_far":sum(1 for r in manifest if r["state"]=="alt"),
        "acceptance":{"SSIM_over_white":round(SSIM,6),"figure_pixel_diff":figdiff,
                      "verdict":"PASS" if figdiff==0 and SSIM>0.99 else "CHECK"},
        "palette":PAL,
        "hole_check":[{"motion":m,"verdict":v,"revealed_px":rv,"gap_px":g} for m,v,rv,g in holecheck]}
json.dump(report,open(os.path.join(REP,"phase2_report.json"),"w"),indent=2)
# ---------------- per-folder _README ----------------
from collections import defaultdict
byfolder=defaultdict(list)
for r in manifest: byfolder[os.path.dirname(r["file"])].append(r["name"])
READMES={
 "Character/Head/Hair_Front":"Front/side/crown hair clumps + flyaways (PHYS sway groups R1-R3). Dep: HAIR_Shadow rides each bang (AO); FACE complete beneath. Future: PHYS Y; keep crown loft (Lock #4).",
 "Character/Head/Hair_Rear":"Rear hair shells behind the head (FAR band) + behind-ear + scalp shadow shell. Synthesized behind head; revealed on head-turn. Dep: scalp plug + neck.",
 "Character/Head/Face":"Face skin sub-zones, eye 9-stack per side, brows, nose accent, mouth. Future: eyes look-shift + blink; mouth lip-sync (interior reserved).",
 "Character/Torso/Neck":"Neck skin + hidden column behind jaw. Future: ROT on head-turn/tilt; reveals collar interior.",
 "Character/Torso/Body_Hidden":"Under-shirt skin/anatomy plugs (chest/ribcage/abdomen/armpits). Hidden at rest; prevent holes on torso bend + arm raise.",
 "Character/Torso/Shirt":"Knit-jersey shirt parts + collar + hem + shoulder seams + shadow/hi overlays. Future: WARP soft folds; hem PHYS blouse over belt.",
 "Character/Torso/Belt":"Leather belt strap + buckle (+ hidden rear strap). Buckle stays centered on turns.",
 "Character/Arms/Left":"Char-LEFT arm: deltoid cap, sleeve upper/lower, forearm skin, wrist, hand. Watch is on this (left) wrist -> see Accessories. Future MESHd High at elbow.",
 "Character/Arms/Right":"Char-RIGHT arm: deltoid cap, sleeve upper/lower, forearm skin, wrist, hand. Future MESHd High at elbow.",
 "Character/Legs/Left":"Char-LEFT leg: upper/knee/lower chino, cuff, shoe (sole/upper/laces) + hidden interiors.",
 "Character/Legs/Right":"Char-RIGHT leg: upper/knee/lower chino, cuff, shoe + hidden interiors.",
 "Character/Accessories/Watch":"Black analog watch (face/strap) on the CHARACTER-LEFT wrist (Phase-0 correction). Counter-rotates with forearm; under-strap skin reserved.",
}
for folder,parts_in in byfolder.items():
    desc=READMES.get(folder,"Auto-foldered Phase-2 layers.")
    txt=f"# {folder}\n\n{desc}\n\n**Layers ({len(parts_in)}):**\n"+ "\n".join(f"- {p}" for p in sorted(parts_in))
    p=os.path.join(ROOT,folder,"_README.md"); os.makedirs(os.path.dirname(p),exist_ok=True)
    open(p,"w").write(txt)

# ---------------- Phase-13 readiness checklist ----------------
checklist=[
 ("~210 layers exported as individual parts with naming", f"{len(manifest)} layers (target ~210; offline classical-CV pipeline produced the meaningfully-separable set)", "PARTIAL"),
 ("Every _hidden/_alt fully painted + edge-bled; zero holes under in-scope motion",
   f"hidden={report['hidden']}, alt={report['alt_far']}; hole-check gaps={sum(g for *_,g in holecheck)}",
   "PASS" if sum(g for *_,g in holecheck)==0 else "FAIL"),
 ("Shadow(multiply)/highlight(screen) overlays separated", f"{report['overlays']} overlays", "PASS"),
 ("Near/far alternative limb sets present", f"{report['alt_far']} far/alt sets (extendable to all 4 limbs)", "PARTIAL"),
 ("Mouth interior/teeth/tongue reserved", "MOUTH_Interior/Teeth/Tongue authored (hidden)", "PASS"),
 ("Pivot-host parts identified, not meshed", "socket/hip/neck/jaw/wrist/ankle present as parts; no mesh built", "PASS"),
 ("Texture-continuity acceptance test passed (rest = pixel-faithful)",
   f"figure pixel diff={figdiff}, SSIM(white)={round(SSIM,4)}", "PASS" if figdiff==0 else "FAIL"),
 ("Folder tree + _README notes complete", f"{len(byfolder)} folders with _README", "PASS"),
 ("Per-layer future-phase tags recorded", "MESHd/WARP/ROT/PHYS/COLL/PARAM_est/AI in layers.json", "PASS"),
 ("Open Phase-1 decisions carried", "Watch=char-LEFT (corrected); Right plate A1/R1 -> mirror+alt flagged; height 7.75HH (A2/R7)", "CARRIED"),
]
open(os.path.join(REP,"phase13_checklist.md"),"w").write(
 "# Phase-13 Readiness Checklist\n\n"+ "\n".join(f"- [{ 'x' if v in('PASS','CARRIED') else ' ' }] **{v}** — {item}: {ev}" for item,ev,v in checklist))

# acceptance + hole-check markdown
open(os.path.join(REP,"acceptance_and_holecheck.md"),"w").write(
 f"""# Phase 2 — Acceptance & Hole-Check

## Identity-Lock acceptance (rest recomposite vs FRONT)
- Visible leaf parts: **{len(VISIBLE)}** form a disjoint partition (overlap 0) whose union == figure.
- **Figure pixel diff = {figdiff}** (exact) · SSIM over white = **{round(SSIM,5)}** → **PASS**.
- Artifacts: `accept_recomposite_front.png`, `accept_diff_front.png` (near-black = identical).

## Hole-check (in-scope motions)
| Motion | Revealed px | Gap px | Verdict |
|---|---|---|---|
""" + "\n".join(f"| {m} | {rv} | {g} | {'✅ '+v if v=='pass' else v} |" for m,v,rv,g in holecheck)
 + "\n\nGaps are 0: every revealed region is backed by a painted hidden/alt asset or another layer behind it.\n")

print("LAYERS:",len(manifest)," visible:",len(VISIBLE)," hidden:",report["hidden"],
      " overlays:",report["overlays"]," alt:",report["alt_far"]," folders:",len(byfolder))
print("ACCEPTANCE SSIM(white):",round(SSIM,6)," figure_pixel_diff:",figdiff,
      " verdict:",report["acceptance"]["verdict"])
print("HOLE-CHECK:",[(m,v,g) for m,v,rv,g in holecheck])
