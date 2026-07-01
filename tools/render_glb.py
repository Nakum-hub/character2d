#!/usr/bin/env python3
"""Minimal offline software renderer for a GLB mesh (rest/bind pose).
Front + side orthographic, z-buffered, Lambert normal shading. numpy only."""
import sys, os, json, struct, numpy as np
from PIL import Image

CT = {5120:('b',1),5121:('B',1),5122:('h',2),5123:('H',2),5125:('I',4),5126:('f',4)}
NC = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}

def read_glb(path):
    d=open(path,'rb').read(); off=12; g=None; b=None
    L=struct.unpack_from('<I',d,8)[0]
    while off<L:
        clen,ctype=struct.unpack_from('<II',d,off); off+=8
        c=d[off:off+clen]; off+=clen
        if ctype==0x4E4F534A: g=json.loads(c.decode())
        elif ctype==0x004E4942: b=c
    return g,b

def accessor(g,b,idx):
    a=g['accessors'][idx]; bv=g['bufferViews'][a['bufferView']]
    fmt,sz=CT[a['componentType']]; nc=NC[a['type']]
    base=bv.get('byteOffset',0)+a.get('byteOffset',0)
    stride=bv.get('byteStride') or sz*nc
    out=np.empty((a['count'],nc),dtype=np.float64 if fmt=='f' else np.int64)
    for i in range(a['count']):
        o=base+i*stride
        out[i]=struct.unpack_from('<'+fmt*nc,b,o)
    return out

def rasterize(V,N,tris,W=700,H=1000,view='front'):
    P=V.copy()
    if view=='side':  # rotate 90 about Y
        x=P[:,0].copy(); z=P[:,2].copy(); P[:,0]=z; P[:,2]=-x
        Nn=N.copy(); nx=Nn[:,0].copy(); nz=Nn[:,2].copy(); Nn[:,0]=nz; Nn[:,2]=-nx
    else: Nn=N
    mn=P.min(0); mx=P.max(0); ctr=(mn+mx)/2; span=(mx-mn).max()*1.12
    sx=(P[:,0]-ctr[0])/span*W + W/2
    sy=H - ((P[:,1]-ctr[1])/span*W + H/2)   # flip Y
    sz=(P[:,2]-ctr[2])/span
    img=np.ones((H,W,3),np.float32)*0.09
    zb=np.full((H,W),1e9)
    L=np.array([0.3,0.4,1.0]); L=L/np.linalg.norm(L)
    for a,bb,c in tris:
        xs=[sx[a],sx[bb],sx[c]]; ys=[sy[a],sy[bb],sy[c]]
        minx=max(int(min(xs)),0); maxx=min(int(max(xs))+1,W)
        miny=max(int(min(ys)),0); maxy=min(int(max(ys))+1,H)
        if minx>=maxx or miny>=maxy: continue
        x0,y0=sx[a],sy[a]; x1,y1=sx[bb],sy[bb]; x2,y2=sx[c],sy[c]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-9: continue
        nrm=(Nn[a]+Nn[bb]+Nn[c]); ln=np.linalg.norm(nrm)
        if ln<1e-9: continue
        nrm/=ln
        sh=max(0.0,float(nrm@L))*0.8+0.22
        zc=(sz[a]+sz[bb]+sz[c])/3
        yy,xx=np.mgrid[miny:maxy,minx:maxx]
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
        w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den
        w2=1-w0-w1
        m=(w0>=0)&(w1>=0)&(w2>=0)
        if not m.any(): continue
        sub=zb[miny:maxy,minx:maxx]
        closer=m&(zc<sub)
        sub[closer]=zc
        col=np.array([sh*0.75,sh*0.8,sh*0.9])
        img[miny:maxy,minx:maxx][closer]=col
    return (np.clip(img,0,1)*255).astype(np.uint8)

def main():
    path=sys.argv[1]; out=sys.argv[2]
    g,b=read_glb(path)
    prim=g['meshes'][0]['primitives'][0]
    V=accessor(g,b,prim['attributes']['POSITION'])
    N=accessor(g,b,prim['attributes']['NORMAL']) if 'NORMAL' in prim['attributes'] else np.zeros_like(V)
    idx=accessor(g,b,prim['indices']).reshape(-1).astype(int)
    tris=idx.reshape(-1,3)
    print('verts',len(V),'tris',len(tris))
    for view in ('front','side'):
        im=rasterize(V,N,tris,view=view)
        Image.fromarray(im).save(out.replace('.png',f'_{view}.png'))
        print('wrote',out.replace('.png',f'_{view}.png'))

if __name__=='__main__': main()
