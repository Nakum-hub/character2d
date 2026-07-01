#!/usr/bin/env python3
"""
Build CLEAN runtime assets for the embeddable live model (character2d).
Whole clean parts (Live2D/Spine-style) instead of per-triangle warp tiles:
  back_hair, body, head_base (eyes+mouth inpainted to skin so blink/talk reveal real pixels),
  eyes, mouth. Rendered by web_runtime/index.html with smooth whole-part deformers.
Reproducible: python3 tools/build_runtime_assets.py
"""
import numpy as np, cv2, os, json
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
im=np.array(Image.open(ROOT+"/assets/reference/front.png").convert("RGBA")); H,W=im.shape[:2]
A=im[:,:,3]>40; rgb=im[:,:,:3]; os.makedirs(ROOT+"/web_runtime/assets",exist_ok=True)
NECK=705; BODY_TOP=545; yy,xx=np.mgrid[0:H,0:W]
head_mask=A&(yy<NECK); body_mask=A&(yy>=BODY_TOP)
eye=((yy>438)&(yy<512)&(((xx>678)&(xx<772))|((xx>792)&(xx<884))))
mouth=(yy>548)&(yy<598)&(xx>734)&(xx<826)
def save(arr,name):
    ys,xs=np.where(arr[:,:,3]>8); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    Image.fromarray(arr[y0:y1,x0:x1]).save(ROOT+f"/web_runtime/assets/{name}.png")
    return dict(name=name,x=int(x0),y=int(y0),w=int(x1-x0),h=int(y1-y0))
meta={}
hb=im.copy(); hb[~head_mask,3]=0
hb[:,:,:3]=cv2.inpaint(hb[:,:,:3], ((eye|mouth)&head_mask).astype(np.uint8)*255, 6, cv2.INPAINT_TELEA)
meta['head_base']=save(hb,'head_base')
es=im.copy(); es[~(eye&head_mask),3]=0; meta['eyes']=save(es,'eyes')
ms=im.copy(); ms[~(mouth&head_mask),3]=0; meta['mouth']=save(ms,'mouth')
bd=im.copy(); bd[~body_mask,3]=0; meta['body']=save(bd,'body')
hair_col=tuple(int(c) for c in rgb[(yy>320)&(yy<420)&(xx>700)&(xx<860)&(rgb.max(2)<95)].mean(0))
sil=cv2.dilate(head_mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=10)>0
bh=np.zeros((H,W,4),np.uint8); bh[sil,:3]=hair_col; bh[sil,3]=255; bh=cv2.GaussianBlur(bh,(0,0),3)
meta['back_hair']=save(bh,'back_hair')
meta['pivots']=dict(neck=[778,660],chest=[783,950],hip=[780,1630],eyeL=[838,472],eyeR=[725,472],mouth=[780,572])
meta['canvas']=[W,H]; meta['head_bottom']=NECK
json.dump(meta,open(ROOT+"/web_runtime/assets/layout.json","w"),indent=1)
print("runtime assets built:",[k for k in meta if k not in('pivots','canvas','head_bottom')])
