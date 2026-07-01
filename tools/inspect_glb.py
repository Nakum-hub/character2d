#!/usr/bin/env python3
"""Offline GLB inspector: dumps glTF JSON structure (nodes/skins/animations/materials)
and extracts embedded images. No external deps (numpy only for accessor min/max)."""
import sys, os, json, struct

def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a glb"
    off = 12
    gltf = None
    bin_chunk = None
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off+clen]
        off += clen
        if ctype == 0x4E4F534A:      # JSON
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:    # BIN
            bin_chunk = chunk
    return gltf, bin_chunk

def acc_minmax(g, idx):
    a = g["accessors"][idx]
    return a.get("min"), a.get("max"), a.get("count")

def main():
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else None
    g, b = read_glb(path)
    print("="*70)
    print("FILE:", os.path.basename(path))
    print("asset:", g.get("asset", {}))
    for k in ("scenes","nodes","meshes","materials","textures","images","skins","animations","accessors","bufferViews"):
        if k in g: print(f"  {k:14s}: {len(g[k])}")
    # nodes / bones
    nodes = g.get("nodes", [])
    names = [n.get("name","") for n in nodes]
    # skins -> joints
    for si, sk in enumerate(g.get("skins", [])):
        joints = sk.get("joints", [])
        print(f"\nSKIN {si}: {len(joints)} joints, skeleton root node = {sk.get('skeleton')}")
        jnames = [names[j] if j < len(names) else f"?{j}" for j in joints]
        print("  joints:", ", ".join(jnames))
    # animation clips + duration
    for ai, an in enumerate(g.get("animations", [])):
        chans = an.get("channels", [])
        # duration = max of all sampler input maxes
        dur = 0.0
        paths = {}
        for ch in chans:
            tgt = ch.get("target", {}).get("path")
            paths[tgt] = paths.get(tgt, 0) + 1
            samp = an["samplers"][ch["sampler"]]
            mn, mx, cnt = acc_minmax(g, samp["input"])
            if mx: dur = max(dur, mx[0])
        print(f"\nANIM {ai}: '{an.get('name','')}' channels={len(chans)} dur={dur:.3f}s  by-path={paths}")
    # materials (look/shading)
    print("\nMATERIALS:")
    for mi, m in enumerate(g.get("materials", [])):
        pbr = m.get("pbrMetallicRoughness", {})
        print(f"  [{mi}] {m.get('name','')}: baseColor={pbr.get('baseColorFactor')} "
              f"metallic={pbr.get('metallicFactor')} rough={pbr.get('roughnessFactor')} "
              f"emissive={m.get('emissiveFactor')} doubleSided={m.get('doubleSided')} "
              f"alphaMode={m.get('alphaMode')} baseTex={'baseColorTexture' in pbr}")
    # extract images
    if outdir and "images" in g:
        os.makedirs(outdir, exist_ok=True)
        for ii, im in enumerate(g["images"]):
            mime = im.get("mimeType","image/png")
            ext = ".png" if "png" in mime else ".jpg"
            if "bufferView" in im:
                bv = g["bufferViews"][im["bufferView"]]
                start = bv.get("byteOffset",0); ln = bv["byteLength"]
                blob = b[start:start+ln]
            else:
                continue
            fn = os.path.join(outdir, f"img{ii}_{im.get('name','tex')}{ext}")
            with open(fn,"wb") as f: f.write(blob)
            print(f"  wrote {fn} ({ln} bytes)")

if __name__ == "__main__":
    main()
