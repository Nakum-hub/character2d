#!/usr/bin/env python3
"""
Milestone 1 verifier — FIDELITY-FIRST layers. Renders the EXACT transforms the web runtime
uses, over a pose sweep + torture test, to prove dimensional turn + zero tearing.
Head is intact (perfect); depth = front-hair parallax + iris parallax/gaze + head turn.
Blink = clean skin lid drawn over the eye. numpy+cv2+PIL. Mirrors web_runtime math 1:1.
"""
import numpy as np, cv2, os, json, math
from PIL import Image, ImageDraw
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC=os.path.join(ROOT,"web_runtime","assets"); L=json.load(open(os.path.join(SRC,"layout.json")))
CW,CH=L["canvas"]; NECK=L["pivots"]["neck"]; EY=L["eyes"]

def full(name):
    p=L[name]; im=np.array(Image.open(os.path.join(SRC,name+".png")).convert("RGBA"))
    c=np.zeros((CH,CW,4),np.uint8); c[p["y"]:p["y"]+p["h"],p["x"]:p["x"]+p["w"]]=im; return c
LAYERS={n:full(n) for n in ["back_hair","body","head","iris","front_hair"]}

def Fwd(piv,rot,sx,sy,shear,tx,ty):
    c,s=math.cos(rot),math.sin(rot)
    Lin=np.array([[c,-s],[s,c]])@np.array([[1,shear],[0,1]])@np.array([[sx,0],[0,sy]])
    piv=np.array(piv,float); t=np.array([tx,ty],float); off=piv+t-Lin@piv
    return np.array([[Lin[0,0],Lin[0,1],off[0]],[Lin[1,0],Lin[1,1],off[1]],[0,0,1]])
def warpF(layer,F3):
    return cv2.warpAffine(layer,cv2.invertAffineTransform(F3[:2]),(CW,CH),flags=cv2.INTER_LINEAR)
def over(dst,src):
    a=src[:,:,3:4].astype(np.float32)/255.0
    dst[:,:,:3]=(src[:,:,:3]*a+dst[:,:,:3]*(1-a)).astype(np.uint8)
    dst[:,:,3]=np.clip(dst[:,:,3].astype(int)+src[:,:,3],0,255).astype(np.uint8); return dst

def head_F(yaw,pitch,bob,extra_tx=0,extra_ty=0,rk=0):
    return Fwd(NECK,0.05*yaw+0.02*pitch+rk,(1-0.12*abs(yaw)),1+0.05*pitch,0.05*yaw,24*yaw+extra_tx,26*pitch+bob+extra_ty)

def draw_lids(canvas,yaw,pitch,bob,blink):
    if blink<=0.01: return canvas
    F=head_F(yaw,pitch,bob)  # lids ride the head
    col=tuple(EY["lidskin"]); rx=EY["rx"]+3
    img=Image.fromarray(canvas); d=ImageDraw.Draw(img)
    for c in (EY["R"],EY["L"]):
        # transform eye centre through head forward transform
        p=F@np.array([c[0],c[1]-EY["ry"],1.0]); topx,topy=p[0],p[1]
        cover=blink*(2*EY["ry"]+4)
        d.ellipse([topx-rx,topy-4,topx+rx,topy+cover],fill=col)
        if blink>0.6:  # subtle closed-lash crease
            d.line([topx-rx+3,topy+cover-2,topx+rx-3,topy+cover-2],fill=(90,60,50),width=2)
    return np.array(img)

def render(yaw=0,pitch=0,blink=0,gazeX=0,gazeY=0,breath=0):
    bob=2.4*breath
    canvas=np.zeros((CH,CW,4),np.uint8)
    over(canvas,warpF(LAYERS["back_hair"],Fwd(NECK,0.05*yaw,1-0.10*abs(yaw),1,0.04*yaw,16*yaw,bob*0.7)))
    over(canvas,warpF(LAYERS["body"],Fwd(L["pivots"]["hip"],0.004*math.sin(breath*math.pi),1,1+0.006*breath,0,0,bob)))
    over(canvas,warpF(LAYERS["head"],head_F(yaw,pitch,bob)))
    # iris: head + feature parallax + gaze (clamped small so it stays on the baked eye)
    gx=max(-1,min(1,gazeX)); gy=max(-1,min(1,gazeY))
    over(canvas,warpF(LAYERS["iris"],head_F(yaw,pitch,bob,extra_tx=7*yaw+4.5*gx,extra_ty=4*pitch+3.5*gy)))
    # blink lids (over eyes, under front hair)
    canvas=draw_lids(canvas,yaw,pitch,bob,blink)
    # front hair parallax (moves most -> depth) + tiny sway
    over(canvas,warpF(LAYERS["front_hair"],head_F(yaw,pitch,bob,extra_tx=6*yaw,rk=0.03*yaw)))
    return canvas

def onbg(canvas,crop=(575,255,985,720),bg=(28,30,38)):
    a=canvas[:,:,3:4].astype(np.float32)/255.0
    out=(canvas[:,:,:3]*a+np.array(bg)*(1-a)).astype(np.uint8)
    x0,y0,x1,y1=crop; return out[y0:y1,x0:x1]

if __name__=="__main__":
    poses=[("neutral",{}),("look L",{"yaw":-0.7,"gazeX":-0.7}),("look R",{"yaw":0.7,"gazeX":0.7}),
           ("look up",{"pitch":-0.6,"gazeY":-0.6}),("look down",{"pitch":0.6,"gazeY":0.6}),
           ("blink",{"blink":0.95}),("gaze L",{"gazeX":-1}),("breath",{"breath":1})]
    tiles=[(n,Image.fromarray(onbg(render(**pp)))) for n,pp in poses]
    w,h=tiles[0][1].size; sheet=Image.new('RGB',(w*4,h*2),(15,15,18))
    for i,(n,im) in enumerate(tiles):
        sheet.paste(im,((i%4)*w,(i//4)*h)); ImageDraw.Draw(sheet).text(((i%4)*w+6,(i//4)*h+4),n,fill=(94,234,212))
    sheet.save(os.path.join(ROOT,"Character/_reports/m1/posesweep.png"),quality=91); print("posesweep",sheet.size)
    tor=[("yaw -1",{"yaw":-1}),("yaw +1",{"yaw":1}),("yaw+1 pitch+1",{"yaw":1,"pitch":1}),("yaw-1 pitch-1 blink",{"yaw":-1,"pitch":-1,"blink":1})]
    w2,h2=tiles[0][1].size; sheet2=Image.new('RGB',(w2*4,h2),(15,15,18))
    for i,(n,pp) in enumerate(tor):
        sheet2.paste(Image.fromarray(onbg(render(**pp))),(i*w2,0)); ImageDraw.Draw(sheet2).text((i*w2+6,4),n,fill=(255,120,120))
    sheet2.save(os.path.join(ROOT,"Character/_reports/m1/torture.png"),quality=91); print("torture",sheet2.size)
