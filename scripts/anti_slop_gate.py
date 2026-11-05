#!/usr/bin/env python3
"""
ANTI-SLOP MECHANICAL GATE
Fails with exit code 1 if the AI agent attempts UI theater, fake certifications,
image-swapping disguised as 3D CAD, or disconnected backend-to-frontend data.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_EXTS = {".html", ".js", ".jsx", ".ts", ".tsx"}

# 1. Banned marketing/fraudulent certification buzzwords
BANNED_CLAIMS = [
    r"\bASME\s+BPVC\s+CERTIFIED\b",
    r"\bSIL-[1234]\s+CERTIFIED\b",
    r"\bTÜV\s+CERTIFIED\b",
    r"\bISO\s+\d+\s+CERTIFIED\b",
    r"\bU-STAMP\s+CERTIFIED\b",
    r"\bCERTIFIED\b",
    r"\bGUARANTEED\b",
    r"\bU-STAMP\b",
    r"\bSTAMPED\b",
]

# 2. Required WebGL/3D primitives if any file claims "3D" or "CAD"
REQUIRED_3D_PRIMITIVES = [
    r"THREE\.Scene|new\s+Scene\s*\(",
    r"THREE\.Raycaster|new\s+Raycaster\s*\(",
    r"OrbitControls",
    r"WebGLRenderer",
]

# 3. Anti-Theater: Detect fake image-swapping inside "3D/CAD" modules
THEATER_PATTERNS = [
    (r"\.src\s*=\s*['\"].*\.(png|jpg|jpeg|webp)", "Image `.src` swapping detected (Possible fake 3D viewer)"),
    (r"href=['\"]#['\"]", "Dead placeholder link `href='#'` detected"),
    (r"TODO|FIXME|PLACEHOLDER|coming soon", "Unfinished placeholder marker detected"),
]


def scan_files():
    errors = []
    frontend_files = [
        p for p in ROOT.rglob("*")
        if p.suffix in FRONTEND_EXTS
        and "node_modules" not in p.parts
        and ".git" not in p.parts
        and "vendor" not in p.parts
    ]

    if not frontend_files:
        errors.append("CRITICAL: No frontend files found to audit.")
        return errors

    full_frontend_Source = ""
    claims_3d_cad = False

    for file_path in frontend_files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        full_frontend_Source += "\n" + content
        rel_path = file_path.relative_to(ROOT)

        # Check A: Fraudulent Certification Claims
        for pattern in BANNED_CLAIMS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                errors.append(
                    f"[FRAUDULENT CLAIM] {rel_path}: Found '{match.group(0)}'. "
                    "Replace with exact standard clause + runnable calculation script citation."
                )

        # Check B: Theater & Placeholder Patterns
        for pattern, msg in THEATER_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                errors.append(f"[UI THEATER] {rel_path}: {msg} -> `{match.group(0)}`")

        if re.search(r"3D\s+CAD|Digital\s+Twin|Interactive\s+3D", content, re.IGNORECASE):
            claims_3d_cad = True

    # Check C: If UI claims 3D CAD / Digital Twin, verify real WebGL/Three.js exists
    if claims_3d_cad:
        for prim in REQUIRED_3D_PRIMITIVES:
            if not re.search(prim, full_frontend_Source):
                errors.append(
                    f"[FAKE 3D ENGINE] UI claims '3D CAD / Digital Twin', but required WebGL primitive "
                    f"`{prim}` is missing from frontend code. Static image galleries are banned."
                )

    # Check D: Verify Traceability Manifest Exists and isn't empty
    manifest_candidates = list(ROOT.rglob("*manifest*.json")) + list(ROOT.rglob("*plant_data*.json"))
    if not manifest_candidates:
        errors.append(
            "[DISCONNECTED BACKEND] No JSON engineering manifest found. "
            "Backend Python calculation scripts MUST export a JSON artifact consumed by the frontend."
        )
    else:
        for m in manifest_candidates:
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
                if not data or len(str(data)) < 200:
                    errors.append(f"[SHALLOW MANIFEST] {m.relative_to(ROOT)} is empty or trivially small.")
            except Exception as e:
                errors.append(f"[CORRUPT MANIFEST] {m.relative_to(ROOT)} failed JSON parse: {e}")

    return errors


if __name__ == "__main__":
    print("=" * 72)
    print("RUNNING ANTI-SLOP ENGINEERING GATE...")
    print("=" * 72)
    failures = scan_files()
    if failures:
        print(f"\n[FAIL] {len(failures)} SLOP VIOLATION(S) DETECTED:\n")
        for idx, err in enumerate(failures, 1):
            print(f"  {idx}. {err}")
        print("\nTASK REJECTED. Fix all violations above before yielding turn.")
        sys.exit(1)
    else:
        print("\n[PASS] Zero slop violations detected. Structural gate passed.")
        sys.exit(0)
