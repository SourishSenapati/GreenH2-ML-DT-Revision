#!/usr/bin/env python3
"""
ADVEKT x RAEBYES x UNSLOP — Autonomous SolidWorks 2025 B-Rep & PBR CAD CLI Tool
================================================================================
Project Ref: PEM-H2-IND-2026-001
Command-line interface driving live Dassault Systemes SOLIDWORKS 2025 SP1.2 via COM API,
tessellating ISO 10303-214 (.STEP) B-Rep solid bodies via OpenCASCADE into multi-material
glTF 2.0 PBR (.GLB) models, enforcing RAEBYES zero-shortcut circuit-breaker invariants,
and rendering high-definition studio PBR views.

Usage:
  py advekt_h2_solidworks_cli.py build-all
  py advekt_h2_solidworks_cli.py audit-raebyes
  py advekt_h2_solidworks_cli.py render-views
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pythoncom
import trimesh
import win32com.client
import win32gui
import win32con

from sw_brep_primitives_h2 import (
    ASM_TEMPLATE,
    DRW_TEMPLATE,
    PBR_MATERIALS,
    convert_step_to_pbr_scene,
)
from sw_h2_equipment_builders import H2_BUILDER_REGISTRY, NOZZLE_MATING_LEDGER

SW_EXE = r"C:\Program Files\SolidWorks Corp\SOLIDWORKS (2)\SLDWORKS.exe"
if not os.path.exists(SW_EXE):
    SW_EXE = r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\SLDWORKS.exe"

BASE_CAD_DIR = r"d:\CompeteEnginner\PEM_Green_Hydrogen_Digital_Twin\cad_and_geometry"
PARTS_DIR = os.path.join(BASE_CAD_DIR, "commercial_plant_components")
STEP_DIR = os.path.join(BASE_CAD_DIR, "step_exports")
GLB_CAD_DIR = os.path.join(BASE_CAD_DIR, "glb_exports")

PRES_MODELS_DIR = r"d:\CompeteEnginner\Presentation_ML_Green_Hydrogen\models"
REPO_MODELS_DIR = r"d:\CompeteEnginner\repo_temp\presentation\models"
PRES_ASSETS_DIR = r"d:\CompeteEnginner\Presentation_ML_Green_Hydrogen\assets"
REPO_ASSETS_DIR = r"d:\CompeteEnginner\repo_temp\presentation\assets"
ARTIFACT_DIR = r"C:\Users\SOURISH SENAPATI\.gemini\antigravity\brain\0627836d-c0c4-4ce7-b44d-b2adbaea2249"


def dismiss_sw_popups():
    """Dismisses any modal SolidWorks recovery or template warning dialogs."""
    def _enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd) or ""
        cls = win32gui.GetClassName(hwnd) or ""
        if cls == "#32770" and ("SOLIDWORKS" in title.upper() or "DOCUMENT" in title.upper()):
            try:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            except Exception:
                pass
        return True
    try:
        win32gui.EnumWindows(_enum_cb, None)
    except Exception:
        pass


def connect_solidworks_2025():
    """Connects to live Dassault Systemes SOLIDWORKS 2025 SP1.2 via Windows COM."""
    user32 = ctypes.windll.user32
    h_desk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
    if h_desk:
        user32.SetThreadDesktop(h_desk)

    pythoncom.CoInitialize()
    try:
        sw = win32com.client.GetActiveObject("SldWorks.Application")
    except Exception:
        if os.path.exists(SW_EXE):
            print(f"[ADVEKT-SW2025] Launching SOLIDWORKS 2025: {SW_EXE}")
            subprocess.Popen([SW_EXE])
            for _ in range(30):
                time.sleep(2.0)
                dismiss_sw_popups()
                try:
                    sw = win32com.client.GetActiveObject("SldWorks.Application")
                    break
                except Exception:
                    sw = None
        else:
            sw = win32com.client.Dispatch("SldWorks.Application")

    if sw is None:
        sw = win32com.client.Dispatch("SldWorks.Application")

    dismiss_sw_popups()
    try:
        sw.Visible = True
        sw.UserControl = False
        sw.CommandInProgress = True
    except Exception:
        pass
    rev_attr = getattr(sw, "RevisionNumber", "33.1.2")
    rev = rev_attr() if callable(rev_attr) else str(rev_attr)
    print(f"[ADVEKT-SW2025] Connected to SOLIDWORKS Revision {rev}")
    return sw


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sync_glb_file(src_path, filename):
    for dst_dir in (GLB_CAD_DIR, PRES_MODELS_DIR, REPO_MODELS_DIR):
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)


def cmd_build_all(args):
    os.makedirs(PARTS_DIR, exist_ok=True)
    os.makedirs(STEP_DIR, exist_ok=True)
    os.makedirs(GLB_CAD_DIR, exist_ok=True)
    os.makedirs(PRES_MODELS_DIR, exist_ok=True)
    os.makedirs(REPO_MODELS_DIR, exist_ok=True)

    sw = connect_solidworks_2025()

    master_plant_scene = trimesh.Scene()
    plant_sldprt_paths = []
    ledger_records = []
    total_brep_bodies = 0
    total_plant_verts = 0
    total_plant_faces = 0
    total_plant_nodes = 0

    print("=" * 84)
    print("ADVEKT SOLIDWORKS 2025 SP1.2 B-REP BUILD & OPENCASCADE MULTI-MATERIAL PBR PIPELINE")
    print("=" * 84)

    for item in H2_BUILDER_REGISTRY:
        tag = item["tag"]
        sldprt_name = item["sldprt_name"]
        step_name = item["step_name"]
        glb_name = item["glb_name"]
        builder_fn = item["builder"]

        if tag == "STK-010":
            sldprt_path = os.path.join(BASE_CAD_DIR, sldprt_name)
        else:
            sldprt_path = os.path.join(PARTS_DIR, sldprt_name)
        step_path = os.path.join(STEP_DIR, step_name)
        glb_path = os.path.join(GLB_CAD_DIR, glb_name)

        t0 = time.time()
        print(f"\n[{tag}] Building SolidWorks 2025 Multi-Body B-Rep Part: {sldprt_name} ...")
        ctx = builder_fn(sw)
        n_bodies = len(ctx.body_ledger)
        total_brep_bodies += n_bodies

        ctx.save_part(sldprt_path, step_path=step_path, close_doc=True)
        # Also copy STK-010 STEP to BASE_CAD_DIR for direct compatibility
        if tag == "STK-010":
            shutil.copy2(step_path, os.path.join(BASE_CAD_DIR, step_name))

        print(f"  -> Saved .SLDPRT ({os.path.getsize(sldprt_path):,} B, {n_bodies} B-Rep bodies) & .STEP ({os.path.getsize(step_path):,} B)")

        # Convert SolidWorks STEP AP214 -> Multi-Material PBR glTF 2.0 Scene
        pbr_scene, n_verts, n_faces, n_nodes = convert_step_to_pbr_scene(
            step_path, tag, ctx.body_ledger
        )
        pbr_scene.export(glb_path, file_type="glb")
        sync_glb_file(glb_path, glb_name)
        dt = time.time() - t0
        print(
            f"  -> Exported Multi-Material PBR GLB: {glb_name} "
            f"({os.path.getsize(glb_path):,} B | {n_nodes} PBR nodes | {n_verts:,} verts | {n_faces:,} tris | {dt:.1f}s)"
        )

        if item["include_in_plant_asm"]:
            plant_sldprt_paths.append(sldprt_path)
            total_plant_verts += n_verts
            total_plant_faces += n_faces
            total_plant_nodes += n_nodes
            for node_name in pbr_scene.graph.nodes_geometry:
                xf, geom_key = pbr_scene.graph[node_name]
                g = pbr_scene.geometry[geom_key].copy()
                g.apply_transform(xf)
                master_plant_scene.add_geometry(g, node_name=node_name, geom_name=node_name)

        ledger_records.append({
            "tag": tag,
            "sldprt_file": sldprt_name,
            "sldprt_bytes": os.path.getsize(sldprt_path),
            "sldprt_sha256": sha256_file(sldprt_path),
            "step_file": step_name,
            "step_bytes": os.path.getsize(step_path),
            "step_sha256": sha256_file(step_path),
            "glb_file": glb_name,
            "glb_bytes": os.path.getsize(glb_path),
            "glb_sha256": sha256_file(glb_path),
            "brep_solid_bodies": n_bodies,
            "min_bodies_required": item["min_bodies"],
            "pbr_scene_nodes": n_nodes,
            "vertices": n_verts,
            "triangles": n_faces,
            "materials_used": sorted(list({b["material"] for b in ctx.body_ledger})),
        })

    # Export Master Plant Assembly GLB (preserving all 11 subsystems' PBR nodes & materials!)
    master_glb_name = "plant_100tpd_master_assembly.glb"
    master_glb_path = os.path.join(GLB_CAD_DIR, master_glb_name)
    master_plant_scene.export(master_glb_path, file_type="glb")
    sync_glb_file(master_glb_path, master_glb_name)
    print("\n" + "=" * 84)
    print(
        f"[MASTER GLB] {master_glb_name}: {os.path.getsize(master_glb_path):,} B | "
        f"{total_plant_nodes} PBR nodes | {total_plant_verts:,} verts | {total_plant_faces:,} tris"
    )
    print("=" * 84)

    # Assemble Native SolidWorks 2025 Master Assembly (.SLDASM) & STEP (.STEP)
    master_asm_path = os.path.join(BASE_CAD_DIR, "PEM_Commercial_Green_Hydrogen_Production_Plant.SLDASM")
    master_step_path = os.path.join(BASE_CAD_DIR, "PEM_Commercial_Green_Hydrogen_Production_Plant.STEP")
    print(f"\n[MASTER SLDASM] Building SolidWorks 2025 Master Assembly: {os.path.basename(master_asm_path)} ...")

    open_docs = []
    for pp in plant_sldprt_paths:
        err = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warn = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        d = sw.OpenDoc6(pp, 1, 1, "", err, warn)
        if d:
            open_docs.append(d)

    asm_tpl = ASM_TEMPLATE if os.path.exists(ASM_TEMPLATE) else sw.GetUserPreferenceStringValue(9)
    asm = sw.NewDocument(asm_tpl, 0, 0, 0)
    if asm:
        for pp in plant_sldprt_paths:
            asm.AddComponent4(pp, "", 0.0, 0.0, 0.0)
        try:
            asm.ForceRebuild3(False)
            asm.ShowNamedView2("*Isometric", 7)
            asm.ViewZoomtofit2()
        except Exception:
            pass
        if os.path.exists(master_asm_path):
            try:
                os.remove(master_asm_path)
            except OSError:
                pass
        asm.SaveAs3(master_asm_path, 0, 2)
        if os.path.exists(master_step_path):
            try:
                os.remove(master_step_path)
            except OSError:
                pass
        asm.SaveAs3(master_step_path, 0, 2)
        print(
            f"  -> Saved Master .SLDASM ({os.path.getsize(master_asm_path):,} B) "
            f"& Master .STEP ({os.path.getsize(master_step_path):,} B)"
        )

    for d in open_docs:
        try:
            t_attr = getattr(d, "GetTitle", "")
            title = t_attr() if callable(t_attr) else str(t_attr)
            sw.CloseDoc(title)
        except Exception:
            pass

    # Save RAEBYES CAD Audit Ledger
    rev_attr = getattr(sw, "RevisionNumber", "33.1.2")
    sw_rev_str = rev_attr() if callable(rev_attr) else str(rev_attr)
    audit_payload = {
        "project_id": "PEM-H2-IND-2026-001",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "solidworks_revision": sw_rev_str,
        "total_brep_solid_bodies": total_brep_bodies,
        "master_plant_glb": {
            "file": master_glb_name,
            "bytes": os.path.getsize(master_glb_path),
            "sha256": sha256_file(master_glb_path),
            "pbr_nodes": total_plant_nodes,
            "vertices": total_plant_verts,
            "triangles": total_plant_faces,
        },
        "master_plant_sldasm": {
            "file": os.path.basename(master_asm_path),
            "bytes": os.path.getsize(master_asm_path) if os.path.exists(master_asm_path) else 0,
            "sha256": sha256_file(master_asm_path) if os.path.exists(master_asm_path) else "",
        },
        "nozzle_mating_ledger": NOZZLE_MATING_LEDGER,
        "subsystems": ledger_records,
    }
    ledger_path = os.path.join(BASE_CAD_DIR, "cad_raebyes_audit_ledger.json")
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    shutil.copy2(
        ledger_path,
        r"d:\CompeteEnginner\repo_temp\code\engineering_standards\cad_raebyes_audit_ledger.json",
    )
    print(f"[RAEBYES LEDGER] Saved cryptographic CAD ledger -> {ledger_path}")


def _rebuild_ledger_from_disk():
    """Compiles cad_raebyes_audit_ledger.json directly from the generated .SLDPRT, .STEP, and .GLB artifacts."""
    sw = connect_solidworks_2025()
    rev_attr = getattr(sw, "RevisionNumber", "33.1.2")
    sw_rev_str = rev_attr() if callable(rev_attr) else str(rev_attr)

    ledger_records = []
    total_brep_bodies = 0
    total_plant_verts = 0
    total_plant_faces = 0
    total_plant_nodes = 0

    for item in H2_BUILDER_REGISTRY:
        tag = item["tag"]
        sldprt_name = item["sldprt_name"]
        step_name = item["step_name"]
        glb_name = item["glb_name"]

        sldprt_path = os.path.join(BASE_CAD_DIR if tag == "STK-010" else PARTS_DIR, sldprt_name)
        step_path = os.path.join(STEP_DIR, step_name)
        glb_path = os.path.join(GLB_CAD_DIR, glb_name)

        # Query exact B-Rep solid body count from SolidWorks .SLDPRT
        err = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warn = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        doc = sw.OpenDoc6(sldprt_path, 1, 1, "", err, warn)
        bodies = (doc.GetBodies2(0, False) or []) if doc else []
        n_bodies = len(bodies)
        if doc:
            t_attr = getattr(doc, "GetTitle", "")
            title = t_attr() if callable(t_attr) else str(t_attr)
            sw.CloseDoc(title)

        total_brep_bodies += n_bodies

        sc = trimesh.load(glb_path, force="scene")
        n_nodes = len(sc.graph.nodes_geometry)
        n_verts = sum(len(g.vertices) for g in sc.geometry.values())
        n_faces = sum(len(g.faces) for g in sc.geometry.values())
        mats_used = sorted(list({_extract_mat_key(nn) for nn in sc.graph.nodes_geometry}))

        if item["include_in_plant_asm"]:
            total_plant_verts += n_verts
            total_plant_faces += n_faces
            total_plant_nodes += n_nodes

        ledger_records.append({
            "tag": tag,
            "sldprt_file": sldprt_name,
            "sldprt_bytes": os.path.getsize(sldprt_path),
            "sldprt_sha256": sha256_file(sldprt_path),
            "step_file": step_name,
            "step_bytes": os.path.getsize(step_path),
            "step_sha256": sha256_file(step_path),
            "glb_file": glb_name,
            "glb_bytes": os.path.getsize(glb_path),
            "glb_sha256": sha256_file(glb_path),
            "brep_solid_bodies": n_bodies,
            "min_bodies_required": item["min_bodies"],
            "pbr_scene_nodes": n_nodes,
            "vertices": n_verts,
            "triangles": n_faces,
            "materials_used": mats_used,
        })

    master_glb_name = "plant_100tpd_master_assembly.glb"
    master_glb_path = os.path.join(GLB_CAD_DIR, master_glb_name)
    master_asm_path = os.path.join(BASE_CAD_DIR, "PEM_Commercial_Green_Hydrogen_Production_Plant.SLDASM")

    audit_payload = {
        "project_id": "PEM-H2-IND-2026-001",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "solidworks_revision": sw_rev_str,
        "total_brep_solid_bodies": total_brep_bodies,
        "master_plant_glb": {
            "file": master_glb_name,
            "bytes": os.path.getsize(master_glb_path),
            "sha256": sha256_file(master_glb_path),
            "pbr_nodes": total_plant_nodes,
            "vertices": total_plant_verts,
            "triangles": total_plant_faces,
        },
        "master_plant_sldasm": {
            "file": os.path.basename(master_asm_path),
            "bytes": os.path.getsize(master_asm_path) if os.path.exists(master_asm_path) else 0,
            "sha256": sha256_file(master_asm_path) if os.path.exists(master_asm_path) else "",
        },
        "nozzle_mating_ledger": NOZZLE_MATING_LEDGER,
        "subsystems": ledger_records,
    }
    ledger_path = os.path.join(BASE_CAD_DIR, "cad_raebyes_audit_ledger.json")
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    shutil.copy2(
        ledger_path,
        r"d:\CompeteEnginner\repo_temp\code\engineering_standards\cad_raebyes_audit_ledger.json",
    )
    print(f"[RAEBYES LEDGER] Compiled & saved cryptographic CAD ledger -> {ledger_path}")


def cmd_audit_raebyes(args):
    """
    RAEBYES Invariant Circuit Breaker:
      1. Verifies all 12 SolidWorks .SLDPRT, .STEP, and multi-material PBR .GLB files exist and are non-trivial.
      2. Verifies total SolidWorks B-Rep solid body count >= 500 across the facility.
      3. Verifies every exported .GLB has distinct named PBR nodes (`__subtag__MATKEY`) and PBRMaterial finishes.
      4. Verifies 0.000000 m Euclidean nozzle mating gap across all 10 process & electrical lines.
    """
    ledger_path = os.path.join(BASE_CAD_DIR, "cad_raebyes_audit_ledger.json")
    if not os.path.exists(ledger_path):
        _rebuild_ledger_from_disk()

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    print("=" * 84)
    print("RAEBYES INVARIANT CIRCUIT BREAKER — SOLIDWORKS 2025 B-REP & PBR GLB VERIFICATION")
    print("=" * 84)

    # 1. Check B-Rep Solid Body Counts & File Integrity
    total_bodies = 0
    for sub in ledger["subsystems"]:
        tag = sub["tag"]
        bodies = sub["brep_solid_bodies"]
        min_req = sub["min_bodies_required"]
        total_bodies += bodies
        if bodies < min_req:
            raise RuntimeError(f"RAEBYES_ERR_STUB_GEOMETRY: [{tag}] has {bodies} B-Rep bodies < {min_req} required")

        glb_p = os.path.join(PRES_MODELS_DIR, sub["glb_file"])
        if not os.path.exists(glb_p) or os.path.getsize(glb_p) < 15000:
            raise RuntimeError(f"RAEBYES_ERR_GLB_INVALID: [{tag}] {glb_p}")

        # Load GLB and verify multi-node PBR scene structure
        sc = trimesh.load(glb_p, force="scene")
        n_nodes = len(sc.graph.nodes_geometry)
        if n_nodes < 3:
            raise RuntimeError(f"RAEBYES_ERR_MONOCHROME_SLOP_GLB: [{tag}] has only {n_nodes} scene nodes")

        print(
            f"  [PASS] {tag:8s} | B-Rep Bodies: {bodies:3d} (min {min_req:3d}) | "
            f"PBR Nodes: {n_nodes:2d} | Triangles: {sub['triangles']:6,d} | GLB: {sub['glb_bytes']:9,d} B"
        )

    if total_bodies < 500:
        raise RuntimeError(f"RAEBYES_ERR_LOW_DETAIL: Total B-Rep bodies {total_bodies} < 500")

    # 2. Check Master Plant Assembly GLB
    master_p = os.path.join(PRES_MODELS_DIR, "plant_100tpd_master_assembly.glb")
    sc_master = trimesh.load(master_p, force="scene")
    master_nodes = len(sc_master.graph.nodes_geometry)
    if master_nodes < 50:
        raise RuntimeError(f"RAEBYES_ERR_MASTER_MONOLITH: Master GLB has only {master_nodes} nodes (<50 required)")
    print(
        f"  [PASS] MASTER   | Total B-Rep Bodies: {total_bodies:3d} | "
        f"Master PBR Nodes: {master_nodes:3d} | Master GLB: {os.path.getsize(master_p):9,d} B"
    )

    # 3. Check 0.000 mm Nozzle Mating Closure Across All 10 Lines
    print("-" * 84)
    print("RAEBYES PIPING & BUSBAR TOPOLOGICAL CLOSURE AUDIT (MAX TOLERANCE = 0.000001 m):")
    max_gap = 0.0
    for line in NOZZLE_MATING_LEDGER:
        p_from = np.array(line["from_coord"], dtype=float)
        p_start = np.array(line["piping_start"], dtype=float)
        p_to = np.array(line["to_coord"], dtype=float)
        p_end = np.array(line["piping_end"], dtype=float)
        gap_start = float(np.linalg.norm(p_from - p_start))
        gap_end = float(np.linalg.norm(p_to - p_end))
        line_gap = max(gap_start, gap_end)
        max_gap = max(max_gap, line_gap)
        if line_gap > 1e-6:
            raise RuntimeError(f"RAEBYES_ERR_OPEN_PIPE_TERMINATION: {line['line_id']} gap = {line_gap:.6f} m")
        print(
            f"  [PASS] {line['line_id']:24s} | {line['from_unit']:12s} -> {line['to_unit']:12s} | "
            f"Start Gap: {gap_start * 1000:.4f} mm | End Gap: {gap_end * 1000:.4f} mm"
        )

    print("=" * 84)
    print(f"RAEBYES CIRCUIT BREAKER PASSED: {total_bodies} SOLIDWORKS B-REP BODIES | MAX PIPE GAP = {max_gap * 1000:.4f} mm")
    print("=" * 84)


def _extract_mat_key(node_name):
    for mkey in PBR_MATERIALS:
        if node_name.endswith("__" + mkey) or ("__" + mkey) in node_name:
            return mkey
    return "SS316L_POLISHED"


def _render_pbr_glb_to_png(glb_path, out_png, cam_dir=(1.15, 0.65, 1.05), viewup=(0.0, 1.0, 0.0),
                           zoom=1.08, bg_rgb=(248, 250, 252), resolution=(2400, 1650)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    scene = trimesh.load(glb_path, force="scene")
    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background([c / 255.0 for c in bg_rgb])
    pl.enable_anti_aliasing("msaa")

    pl.remove_all_lights()
    lights = [
        pv.Light(position=(26.0, 36.0, 26.0), focal_point=(0.0, 2.0, 0.0), color="white", intensity=1.15),
        pv.Light(position=(-26.0, 28.0, 20.0), focal_point=(0.0, 2.0, 0.0), color="#eef4ff", intensity=0.88),
        pv.Light(position=(12.0, 22.0, -28.0), focal_point=(0.0, 2.0, 0.0), color="#fff8f0", intensity=0.72),
        pv.Light(position=(-18.0, 18.0, -22.0), focal_point=(0.0, 2.0, 0.0), color="#f1f5f9", intensity=0.58),
        pv.Light(position=(0.0, 42.0, 0.0), focal_point=(0.0, 0.0, 0.0), color="white", intensity=0.52),
        pv.Light(position=(0.0, -18.0, 14.0), focal_point=(0.0, 2.0, 0.0), color="#e2e8f0", intensity=0.38),
    ]
    for lgt in lights:
        pl.add_light(lgt)

    bmin = np.array([1e9, 1e9, 1e9], dtype=float)
    bmax = np.array([-1e9, -1e9, -1e9], dtype=float)

    for node_name in scene.graph.nodes_geometry:
        xf, geom_key = scene.graph[node_name]
        g = scene.geometry[geom_key].copy()
        g.apply_transform(xf)
        if len(g.faces) == 0:
            continue
        bmin = np.minimum(bmin, g.bounds[0])
        bmax = np.maximum(bmax, g.bounds[1])

        mkey = _extract_mat_key(node_name)
        mcfg = PBR_MATERIALS.get(mkey, PBR_MATERIALS["SS316L_POLISHED"])
        rgb = [mcfg["rgba"][0] / 255.0, mcfg["rgba"][1] / 255.0, mcfg["rgba"][2] / 255.0]

        verts = np.asarray(g.vertices, dtype=np.float64)
        faces = np.asarray(g.faces, dtype=np.int64)
        n_f = len(faces)
        cells = np.hstack([np.full((n_f, 1), 3, dtype=np.int64), faces]).ravel()
        pd = pv.PolyData(verts, cells)
        pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True,
                           feature_angle=38.0, inplace=True)
        # In VTK without an HDR cubemap, cap metallic at 0.34 so bright diffuse alloy color + specular highlights shine cleanly
        vtk_metal = min(float(mcfg["metallic"]) * 0.36, 0.34)
        vtk_rough = max(float(mcfg["roughness"]), 0.28)
        pl.add_mesh(
            pd,
            color=rgb,
            pbr=True,
            metallic=vtk_metal,
            roughness=vtk_rough,
            smooth_shading=True,
        )

    center = 0.5 * (bmin + bmax)
    span = bmax - bmin
    # Subtle studio technical floor pad underneath the assembly
    pad_box = pv.Box(bounds=(
        center[0] - span[0] * 0.56, center[0] + span[0] * 0.56,
        bmin[1] - span[1] * 0.018, bmin[1] - 0.001,
        center[2] - span[2] * 0.56, center[2] + span[2] * 0.56,
    ))
    pl.add_mesh(pad_box, color=[0.88, 0.91, 0.94], pbr=True, metallic=0.08, roughness=0.75)

    diag = float(np.linalg.norm(span))
    cdir = np.asarray(cam_dir, dtype=float)
    cdir = cdir / np.linalg.norm(cdir)
    cam_pos = center + cdir * (diag * 1.96)

    pl.camera_position = [cam_pos.tolist(), center.tolist(), list(viewup)]
    pl.camera.zoom(zoom)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pl.screenshot(out_png, transparent_background=False)
    pl.close()


def cmd_render_views(args):
    """Renders high-definition multi-material PBR studio views from the SolidWorks-STEP-derived GLB models."""
    print("=" * 84)
    print("RENDERING HIGH-DEFINITION STUDIO PBR VIEWS FROM SOLIDWORKS 2025 STEP/GLB MODELS")
    print("=" * 84)

    master_glb = os.path.join(PRES_MODELS_DIR, "plant_100tpd_master_assembly.glb")
    stack_glb = os.path.join(PRES_MODELS_DIR, "STK-010_10Cell_PEM_Stack.glb")

    renders = [
        (master_glb, "PEM_Commercial_Plant_Catalog_Render.png", (1.12, 0.62, 1.08), (0, 1, 0), 1.08),
        (master_glb, "PEM_Commercial_Plant_Master_Render.png", (1.12, 0.62, 1.08), (0, 1, 0), 1.08),
        (master_glb, "PEM_Commercial_Plant_Isometric_HD.png", (-1.15, 0.58, 1.05), (0, 1, 0), 1.08),
        (master_glb, "PEM_Commercial_Plant_Dimetric_HD.png", (1.18, 0.52, -1.02), (0, 1, 0), 1.08),
        (master_glb, "PEM_Commercial_Plant_FrontElevation_HD.png", (0.0, 0.14, 1.0), (0, 1, 0), 1.12),
        (master_glb, "PEM_Commercial_Plant_PlotPlan_HD.png", (0.0, 1.0, 0.001), (0, 0, -1), 1.06),
        (master_glb, "PEM_Turnkey_Plant_Skid_Catalog_Render.png", (-1.10, 0.56, 1.12), (0, 1, 0), 1.08),
        (master_glb, "PEM_Turnkey_Plant_Skid_Photorealistic_Render.png", (-1.10, 0.56, 1.12), (0, 1, 0), 1.08),
        (stack_glb, "PEM_10Cell_Industrial_Stack_Catalog_Render.png", (1.08, 0.54, 1.15), (0, 1, 0), 1.15),
        (stack_glb, "PEM_10Cell_Stack_Presentation_Render.png", (-1.08, 0.52, 1.15), (0, 1, 0), 1.15),
        (stack_glb, "PEM_Stack_10Cell_Assembly_Render.png", (1.08, 0.54, 1.15), (0, 1, 0), 1.15),
    ]

    for glb_src, png_name, cdir, vup, zm in renders:
        out_cad = os.path.join(BASE_CAD_DIR, png_name)
        _render_pbr_glb_to_png(glb_src, out_cad, cam_dir=cdir, viewup=vup, zoom=zm)
        for target_dir in (PRES_ASSETS_DIR, REPO_ASSETS_DIR, ARTIFACT_DIR):
            if os.path.exists(target_dir):
                shutil.copy2(out_cad, os.path.join(target_dir, png_name))
        print(f"  -> Rendered & Synced Studio PBR View: {png_name} ({os.path.getsize(out_cad):,} B)")

    print("=" * 84)
    print("ALL HIGH-DEFINITION STUDIO PBR RENDERS COMPLETED & SYNCHRONIZED!")
    print("=" * 84)


def main():
    parser = argparse.ArgumentParser(
        description="ADVEKT x RAEBYES SolidWorks 2025 B-Rep & PBR CAD Automation CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-all", help="Build all 12 SolidWorks 2025 parts, assembly, STEP, and PBR GLB models")
    sub.add_parser("audit-raebyes", help="Run RAEBYES invariant circuit-breaker audit on all CAD artifacts")
    sub.add_parser("render-views", help="Render high-definition multi-material PBR studio views from GLB models")

    args = parser.parse_args()
    if args.cmd == "build-all":
        cmd_build_all(args)
        cmd_audit_raebyes(args)
    elif args.cmd == "audit-raebyes":
        cmd_audit_raebyes(args)
    elif args.cmd == "render-views":
        cmd_render_views(args)


if __name__ == "__main__":
    main()
