#!/usr/bin/env python3
"""Render an idle-life loop GIF (gaze drift + breathing + natural blinks + a saccade),
using the SAME transforms as the web runtime, to preview the living head."""
import numpy as np, math, importlib.util, os
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location("m1v", os.path.join(ROOT,"tools/m1_verify.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

N=72; fps=24
sx=sy=gx=gy=0.0   # smoothed
frames=[]; blink=0; next_blink=18
for i in range(N):
    t=i/fps
    # idle gaze target: slow figure-8 + a saccade around frame 30
    tgtx=0.55*math.sin(t*0.9); tgty=0.28*math.sin(t*0.7+1)
    if 30<=i<36: tgtx,tgty=0.9,-0.2      # quick glance
    sx+=(tgtx-sx)*0.12; sy+=(tgty-sy)*0.12
    gx+=(tgtx-gx)*0.22; gy+=(tgty-gy)*0.22   # eyes lead the head
    breath=math.sin(t*1.7)
    # blink schedule (triangular ~110ms)
    next_blink-=1
    if next_blink<=0: blink_t=4; next_blink=int(40+30*np.random.rand())
    try: blink_t
    except NameError: blink_t=0
    if blink_t>0:
        blink=1-abs(blink_t-2)/2; blink_t-=1
    else: blink=0
    canvas=m.render(yaw=sx*0.8,pitch=sy*0.8,blink=max(0,blink),gazeX=gx,gazeY=gy,breath=breath)
    im=Image.fromarray(m.onbg(canvas,crop=(600,258,960,720)))
    im=im.resize((im.width//2,im.height//2))
    frames.append(im)
frames[0].save(os.path.join(ROOT,"Character/_reports/m1/preview.gif"),save_all=True,append_images=frames[1:],
               duration=int(1000/fps),loop=0,disposal=2)
print("gif frames",len(frames),"size",frames[0].size)
