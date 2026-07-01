#!/usr/bin/env python3
"""
Milestone 1 layer builder (character2d 2.5D living head) — FIDELITY-FIRST.
Minimal, non-destructive decomposition so the character stays EXACTLY on-model:
  head       : the original head, fully intact (perfect eyes/brows/nose/mouth/hair baked)
  front_hair : a DUPLICATE of the hair mass, drawn over the identical baked hair -> parallax
               depth with zero fidelity loss and no skin gap on turn
  iris       : slightly-enlarged iris discs that always cover the baked iris -> clean gaze
  back_hair  : tight hair-colour cap behind -> anti-bald safety on turn
  body       : everything below the neck
Blink + eyelids are drawn at runtime from a sampled skin colour (no fragile eye cut).
Reproducible: python3 tools/m1_build.py
"""
import numpy as np, cv2, os, json
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,"web_runtime","assets"); os.makedirs(OUT,exist_ok=True)
im=np.array(Image.open(os.path.join(ROOT,"assets/reference/front.png")).convert("RGBA"))
H,W=im.shape[:2]; rgb=im[:,:,:3].astype(np.int16)
R,G,B=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]; V=rgb.max(2); warmth=R-B; RG=R-G
A=im[:,:,3]>40; yy,xx=np.mgrid[0:H,0:W]; hk=np.ones((3,3),np.uint8)

HEAD_TOP,NECK=268,712
eyeR_c,eyeL_c=(722,489),(837,489); eye_rx,eye_ry=34,20
irisR_c,irisL_c=(722,489),(837,489); iris_r=19          # enlarged: covers baked iris on gaze move
def disc(c,r): return (xx-c[0])**2+(yy-c[1])**2<=r*r
def ell(c,rx,ry): return ((xx-c[0])/rx)**2+((yy-c[1])/ry)**2<=1.0

skinBright=A&(R>150)&(RG>36)&(warmth>48)&(V>150)
shirt=A&(yy>=680)&(V<105)&(warmth<40)
headSil=A&(yy>=HEAD_TOP)&(yy<NECK)&~shirt
headSil=cv2.morphologyEx(headSil.astype(np.uint8),cv2.MORPH_CLOSE,hk,iterations=3).astype(bool)

# hair by region (robust to lit tips)
faceOval=ell((779,548),128,150)
eyeExcl=(ell(eyeR_c,eye_rx+3,eye_ry+3)|ell(eyeL_c,eye_rx+3,eye_ry+3))
hair_core=headSil&(V<128)&(RG<34)&(warmth>=8)&(R>=B-4)
hair_frame=headSil&~skinBright&~faceOval&(V<215)
hair_bangs=headSil&faceOval&(V<122)&(RG<40)
hairM=(hair_core|hair_frame|hair_bangs)&~eyeExcl
hairM=cv2.morphologyEx(hairM.astype(np.uint8),cv2.MORPH_CLOSE,hk,iterations=2)
hairM=cv2.morphologyEx(hairM,cv2.MORPH_OPEN,hk,iterations=1).astype(bool)
n,lab=cv2.connectedComponents(hairM.astype(np.uint8))
if n>1:
    sizes=[(lab==i).sum() for i in range(1,n)]; hairM=lab==(1+int(np.argmax(sizes)))

def save(arr,name):
    ys,xs=np.where(arr[:,:,3]>6); y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
    Image.fromarray(arr[y0:y1,x0:x1]).save(os.path.join(OUT,name+".png"))
    return dict(name=name,x=x0,y=y0,w=x1-x0,h=y1-y0)
def feather(mask,src,blur,grow=0):
    m=mask.astype(np.uint8)*255
    if grow: m=cv2.dilate(m,hk,iterations=grow)
    a=cv2.GaussianBlur(m,(0,0),blur) if blur>0 else m
    return np.dstack([src[:,:,0],src[:,:,1],src[:,:,2],a]).astype(np.uint8)
meta={}

# HEAD — fully intact original (perfect fidelity); bottom (neck) edge faded so there is NO
# hard line over the body when the head turns (the body layer holds the same neck pixels beneath)
fa=cv2.GaussianBlur(headSil.astype(np.uint8)*255,(0,0),1.2).astype(np.float32)
ramp=np.ones((H,1),np.float32); y0r,y1r=628,688
ys=np.arange(y0r,y1r); ramp[y0r:y1r,0]=1-(ys-y0r)/(y1r-y0r); ramp[y1r:,0]=0
fa=(fa*ramp).astype(np.uint8)
head=np.dstack([im[:,:,0],im[:,:,1],im[:,:,2],fa]).astype(np.uint8)
meta['head']=save(head,'head')

# FRONT HAIR — duplicate of the hair mass (crisp), for parallax over the identical baked hair
meta['front_hair']=save(feather(hairM,im[:,:,:3],0.7,0),'front_hair')

# IRIS — enlarged discs (feathered) so gaze offset always covers the baked iris
irisReg=disc(irisR_c,iris_r)|disc(irisL_c,iris_r)
meta['iris']=save(feather(irisReg,im[:,:,:3],0.8,0),'iris')

# BACK HAIR cap — tight hair-colour, subtle
hair_col=tuple(int(c) for c in im[:,:,:3][hairM&(V<110)].mean(0))
cap=cv2.dilate(hairM.astype(np.uint8),hk,iterations=7)
capa=(cv2.GaussianBlur(cap*255,(0,0),3)*0.4).astype(np.uint8)
bh=np.zeros((H,W,4),np.uint8); bh[:,:,0],bh[:,:,1],bh[:,:,2]=hair_col; bh[:,:,3]=capa
meta['back_hair']=save(bh,'back_hair')

# BODY
bodyM=A&(yy>=545)
meta['body']=save(np.dstack([im[:,:,:3],(bodyM.astype(np.uint8)*255)]).astype(np.uint8),'body')

# sampled eyelid skin colour for runtime blink lids (skin just below each brow)
lidskin=tuple(int(c) for c in im[:,:,:3][(yy>468)&(yy<478)&(((xx>700)&(xx<744))|((xx>816)&(xx<860)))&skinBright].mean(0))
meta['pivots']=dict(neck=[779,690],irisR=list(irisR_c),irisL=list(irisL_c),chest=[783,980],hip=[780,1630])
meta['eyes']=dict(R=list(eyeR_c),L=list(eyeL_c),rx=eye_rx,ry=eye_ry,lidskin=list(lidskin))
meta['canvas']=[W,H]; meta['neck']=NECK
json.dump(meta,open(os.path.join(OUT,"layout.json"),"w"),indent=1)
print("built:",[k for k in meta if k not in('pivots','eyes','canvas','neck')],"| lidskin",lidskin)
