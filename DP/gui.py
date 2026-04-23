import os
import math
import traceback
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageTk

PREVIEW_MAX_PIXELS = 1_200_000
PREVIEW_AUTO_SKIP_PIXELS = 4_000_000

DISPLAY_LABELS = {
    "01_house": "01_house",
    "02_double_house": "02_double_house",
    "03_palace": "03_palace",
    "04_palace": "04_palace",
    "04_palace_large": "04_palace_large",
    "11_temple_pyramida": "11_temple_pyramida",
    "12_juego": "12_juego",
    "15_platform_small": "15_platform_small",
    "16_platform_large": "16_platform_large",
    "ruin_small": "ruin_small",
    "ruin_elongated": "ruin_elongated",
    "ruin_large_rectangular": "ruin_large_rectangular",
    "cluster_houses": "cluster_houses",
    "unknown_elongated_building": "unknown_elongated_building",
    "unknown_large_building": "unknown_large_building",
    "unknown_small_building": "unknown_small_building",
    "unknown_elongated_low": "unknown_elongated_low",
    "unknown_large_low": "unknown_large_low",
    "unknown_small_low": "unknown_small_low",
}

COLOR_MAP = {
    "01_house": (0, 255, 0),
    "02_double_house": (0, 180, 0),
    "03_palace": (255, 0, 0),
    "04_palace": (0, 0, 255),
    "04_palace_large": (0, 128, 255),
    "11_temple_pyramida": (255, 255, 0),
    "12_juego": (255, 0, 255),
    "15_platform_small": (0, 255, 255),
    "16_platform_large": (180, 0, 180),
    "ruin_small": (60, 60, 180),
    "ruin_elongated": (100, 100, 200),
    "ruin_large_rectangular": (140, 140, 220),
    "cluster_houses": (144, 238, 144),
    "unknown_elongated_building": (200, 200, 200),
    "unknown_large_building": (170, 170, 170),
    "unknown_small_building": (230, 230, 230),
    "unknown_elongated_low": (120, 120, 120),
    "unknown_large_low": (100, 100, 100),
    "unknown_small_low": (80, 80, 80),
}

from detector_my import Detector


# -----------------------------
# Core processing logic
# -----------------------------

def load_dem(path: str) -> np.ndarray:
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)
        return dem

    if suffix == ".npz":
        data = np.load(path)
        if "dataset" in data:
            return data["dataset"].astype(np.float32)
        first_key = list(data.keys())[0]
        return data[first_key].astype(np.float32)

    if suffix == ".npy":
        return np.load(path).astype(np.float32)

    raise ValueError(f"Nepodporovaný formát DEM: {suffix}")


def load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        mask = src.read(1)
    return (mask > 0).astype(np.uint8)


# notebook flow

def clean_mask_for_recall(mask_bin: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (closed > 0).astype(np.uint8)


def contour_mask(shape, contour):
    tmp = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(tmp, [contour], -1, 1, thickness=-1)
    return tmp


def compute_height_difference(dem_crop, obj_mask, ring_size=3, iterations=2):
    obj_mask_u8 = obj_mask.astype(np.uint8)
    kernel = np.ones((ring_size, ring_size), np.uint8)
    dilated = cv2.dilate(obj_mask_u8, kernel, iterations=iterations)
    ring = ((dilated > 0) & (obj_mask_u8 == 0))

    obj_vals = dem_crop[obj_mask_u8 == 1]
    ring_vals = dem_crop[ring]

    obj_vals = obj_vals[np.isfinite(obj_vals)]
    ring_vals = ring_vals[np.isfinite(ring_vals)]

    if len(obj_vals) == 0:
        return 0.0, np.nan, np.nan

    obj_mean = float(np.mean(obj_vals))
    if len(ring_vals) == 0:
        return 0.0, obj_mean, np.nan

    ring_mean = float(np.mean(ring_vals))
    return float(obj_mean - ring_mean), obj_mean, ring_mean


def contour_features(contour, dem):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(w * h, 1)

    rect = cv2.minAreaRect(contour)
    short_side = min(w, h)
    long_side = max(w, h)
    aspect_ratio = long_side / short_side if short_side > 0 else 999.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / bbox_area if bbox_area > 0 else 0.0

    epsilon = 0.02 * perimeter if perimeter > 0 else 1.0
    approx = cv2.approxPolyDP(contour, epsilon, True)

    contour_local = contour.copy()
    contour_local[:, 0, 0] -= x
    contour_local[:, 0, 1] -= y

    obj_mask = contour_mask((h, w), contour_local)
    dem_crop = dem[y:y + h, x:x + w]
    height_diff, mean_obj_height, mean_ring_height = compute_height_difference(dem_crop, obj_mask)

    return {
        'contour': contour,
        'contour_local': contour_local,
        'rect': rect,
        'bbox': (x, y, w, h),
        'area': float(area),
        'perimeter': float(perimeter),
        'short_side': float(short_side),
        'long_side': float(long_side),
        'aspect_ratio': float(aspect_ratio),
        'solidity': float(solidity),
        'extent': float(extent),
        'approx_vertices': int(len(approx)),
        'height_diff': float(height_diff),
        'mean_obj_height': float(mean_obj_height) if np.isfinite(mean_obj_height) else np.nan,
        'mean_ring_height': float(mean_ring_height) if np.isfinite(mean_ring_height) else np.nan,
        'obj_mask': obj_mask,
    }


def keep_as_candidate(feat, chunked=False):
    w = feat['bbox'][2]
    h = feat['bbox'][3]
    area = feat['area']
    aspect_ratio = feat['aspect_ratio']
    extent = feat['extent']
    solidity = feat['solidity']
    height_diff = feat['height_diff']

    if w < 3 or h < 3:
        return False

    # preserve notebook behavior
    if chunked:
        if area < 30:
            return False
        if aspect_ratio > 12:
            return False
        if solidity < 0.5:
            return False
        if extent < 0.10:
            return False
        if height_diff < -0.5:
            return False
        return True
    else:
        if area < 8:
            return False
        if aspect_ratio > 12:
            return False
        if extent < 0.08 and solidity < 0.2:
            return False
        if height_diff < -0.5:
            return False
        return True


def split_into_chunks_with_coords(dem, mask, n_rows=10, n_cols=10):
    H, W = dem.shape
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)

    chunk_records = []
    for r in range(n_rows):
        for c in range(n_cols):
            row0, row1 = row_edges[r], row_edges[r + 1]
            col0, col1 = col_edges[c], col_edges[c + 1]

            chunk_records.append({
                'chunk_id': len(chunk_records),
                'row0': int(row0),
                'row1': int(row1),
                'col0': int(col0),
                'col1': int(col1),
                'dem': dem[row0:row1, col0:col1],
                'mask': mask[row0:row1, col0:col1],
            })
    return chunk_records


def should_chunk_automatically(shape):
    h, w = shape
    pixels = h * w
    return pixels > 4_000_000 or max(h, w) > 2500


def choose_chunk_grid(shape):
    h, w = shape
    pixels = h * w
    if pixels > 36_000_000 or max(h, w) > 7000:
        return 12, 12
    if pixels > 16_000_000 or max(h, w) > 4500:
        return 10, 10
    if pixels > 4_000_000 or max(h, w) > 2500:
        return 8, 8
    return 6, 6


def normalize_dem_to_bgr(dem: np.ndarray) -> np.ndarray:
    finite = np.isfinite(dem)
    if not finite.any():
        gray = np.zeros(dem.shape, dtype=np.uint8)
    else:
        vals = dem[finite]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(dem, lo, hi)
        gray = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        gray[~finite] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_preview_canvas(dem: np.ndarray, max_dim: int = 1800):
    h, w = dem.shape
    scale = min(max_dim / max(h, w), 1.0)
    preview = normalize_dem_to_bgr(dem)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        preview = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return preview, scale



def get_legend_items(houses):
    labels = sorted({hinfo[2] for hinfo in houses if len(hinfo) > 2})
    return [(label, COLOR_MAP.get(label, (255, 255, 255))) for label in labels]


def draw_rotated_house(preview: np.ndarray, center_x: float, center_y: float, width: float, height: float, angle_deg: float, color, scale: float):
    # OpenCV boxPoints expects ((cx, cy), (w, h), angle)
    rect = ((center_x * scale, center_y * scale), (max(width * scale, 1.0), max(height * scale, 1.0)), float(angle_deg))
    box = cv2.boxPoints(rect)
    box = np.int32(np.round(box))
    thickness = 1 if scale < 0.75 else 2
    cv2.polylines(preview, [box], True, color, thickness)
    return preview


def draw_houses_on_preview(preview: np.ndarray, houses, house_cord, scale: float):
    for hinfo in houses:
        center_x, center_y, label, angle_deg, _, rw, rh, idx, _ = hinfo
        color = COLOR_MAP.get(label, (255, 255, 255))

        if rw is None or rh is None or rw == 0 or rh == 0:
            x, y, w, h = house_cord[idx]
            x1 = int(round(x * scale))
            y1 = int(round(y * scale))
            x2 = int(round((x + w) * scale))
            y2 = int(round((y + h) * scale))
            thickness = 1 if scale < 0.75 else 2
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        else:
            draw_rotated_house(preview, center_x, center_y, rw, rh, angle_deg, color, scale)

    return preview



def get_rotated_box(cx, cy, w, h, angle_deg):
    angle_rad = math.radians(angle_deg)
    dx = w / 2.0
    dy = h / 2.0
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    pts = []
    for x, y in corners:
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        pts.append((cx + rx, cy + ry))
    return pts


def houses_to_dataframe(houses, house_cord):
    rows = []
    for h in houses:
        idx = h[7]
        x, y, w, h_box = house_cord[idx]
        rows.append({
            "center_x": h[0],
            "center_y": h[1],
            "x": x,
            "y": y,
            "w": w,
            "h": h_box,
            "label": h[2],
            "orientation": h[3],
            "avg_height": h[4],
            "orig_index": idx,
        })
    return pd.DataFrame(rows)


def houses_to_wkt_dataframe(houses):
    rows = []
    for h in houses:
        idx = h[7]
        w = h[5]
        h_box = h[6]
        cx = h[0]
        cy = h[1]
        angle = h[3]

        box = get_rotated_box(cx, cy, w, h_box, angle)
        coords = list(box) + [box[0]]
        wkt = "POLYGON((" + ", ".join([f"{pt[0]} {pt[1]}" for pt in coords]) + "))"
        rows.append({
            "wkt": wkt,
            "label": h[2],
            "orientation": angle,
            "avg_height": h[4],
            "orig_index": idx,
        })
    return pd.DataFrame(rows)


def run_pipeline(dem_path, mask_path, output_dir, chunk_mode="auto", force_rows=10, force_cols=10, logger=None):
    def log(msg):
        if logger:
            logger(msg)

    detector = Detector()

    log("Načítavam DEM...")
    dem = load_dem(dem_path)
    log("Načítavam masku...")
    mask_bin = load_mask(mask_path)

    if dem.shape != mask_bin.shape:
        raise ValueError(f"DEM a maska nemajú rovnaké rozmery: {dem.shape} vs {mask_bin.shape}")

    H, W = dem.shape
    log(f"Rozmery vstupu: {H} x {W}")

    if chunk_mode == "yes":
        use_chunks = True
        n_rows, n_cols = int(force_rows), int(force_cols)
    elif chunk_mode == "no":
        use_chunks = False
        n_rows, n_cols = 1, 1
    else:
        use_chunks = should_chunk_automatically(dem.shape)
        n_rows, n_cols = choose_chunk_grid(dem.shape) if use_chunks else (1, 1)

    log(f"Chunkovanie: {'áno' if use_chunks else 'nie'}")
    if use_chunks:
        log(f"Grid chunkov: {n_rows} x {n_cols}")

    if use_chunks:
        chunk_records = split_into_chunks_with_coords(dem, mask_bin, n_rows=n_rows, n_cols=n_cols)
        all_candidate_features = []
        chunk_stats = []

        for rec in chunk_records:
            cleaned_mask = clean_mask_for_recall(rec['mask'])
            contours, _ = cv2.findContours((cleaned_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            local_candidates = []
            for cnt in contours:
                feat = contour_features(cnt, rec['dem'])
                if keep_as_candidate(feat, chunked=True):
                    feat['chunk_id'] = rec['chunk_id']
                    feat['row0'] = rec['row0']
                    feat['col0'] = rec['col0']
                    local_candidates.append(feat)
            all_candidate_features.extend(local_candidates)
            chunk_stats.append({
                'chunk_id': rec['chunk_id'],
                'row0': rec['row0'],
                'col0': rec['col0'],
                'n_contours': len(contours),
                'n_candidates': len(local_candidates),
            })
        chunk_stats_df = pd.DataFrame(chunk_stats)
    else:
        cleaned_mask = clean_mask_for_recall(mask_bin)
        contours, _ = cv2.findContours((cleaned_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_candidate_features = []
        for cnt in contours:
            feat = contour_features(cnt, dem)
            if keep_as_candidate(feat, chunked=False):
                all_candidate_features.append(feat)
        chunk_stats_df = pd.DataFrame([{
            'chunk_id': 0,
            'row0': 0,
            'col0': 0,
            'n_contours': len(contours),
            'n_candidates': len(all_candidate_features),
        }])

    log(f"Počet candidate features: {len(all_candidate_features)}")

    house = []
    house_height = []
    house_size = []
    house_cord = []
    rectangles_for_angles = []

    if use_chunks:
        chunks_by_id = {rec['chunk_id']: rec for rec in chunk_records}
        for feat in all_candidate_features:
            x, y, w, h = feat['bbox']
            row0 = feat['row0']
            col0 = feat['col0']
            chunk_id = feat['chunk_id']

            global_x = col0 + x
            global_y = row0 + y
            obj_mask_crop = feat['obj_mask']
            dem_chunk = chunks_by_id[chunk_id]['dem']
            dem_crop = dem_chunk[y:y + h, x:x + w]

            if obj_mask_crop.size == 0 or dem_crop.size == 0:
                continue
            if obj_mask_crop.shape != dem_crop.shape:
                continue

            house.append(obj_mask_crop.astype(np.uint8))
            house_height.append(dem_crop.astype(np.float32))
            house_size.append([int(w), int(h)])
            house_cord.append([int(global_x), int(global_y), int(w), int(h)])
            rectangles_for_angles.append(feat['rect'])
    else:
        for feat in all_candidate_features:
            x, y, w, h = feat['bbox']
            obj_mask_crop = feat['obj_mask']
            dem_crop = dem[y:y + h, x:x + w]

            if obj_mask_crop.size == 0 or dem_crop.size == 0:
                continue
            if obj_mask_crop.shape != dem_crop.shape:
                continue

            house.append(obj_mask_crop.astype(np.uint8))
            house_height.append(dem_crop.astype(np.float32))
            house_size.append([int(w), int(h)])
            house_cord.append([int(x), int(y), int(w), int(h)])
            rectangles_for_angles.append(feat['rect'])

    log(f"Počet house segmentov: {len(house)}")

    difference = detector.compute_difference(house, house_height)
    orientation = detector.determine_building_orientations(rectangles_for_angles)

    log("Prebieha klasifikácia objektov...")
    houses = detector.find_buildings(None, house, orientation, house_size, house_cord, difference, house_height)
    counts = Counter([h[2] for h in houses]) if houses else Counter()
    log(f"Počet finálnych houses: {len(houses)}")
    log(f"Triedy: {dict(counts)}")

    total_pixels = int(dem.shape[0] * dem.shape[1])
    preview = None
    preview_scale = 1.0

    if total_pixels <= PREVIEW_AUTO_SKIP_PIXELS:
        log("Pripravujem preview obrázok...")
        preview, preview_scale = build_preview_canvas(dem, max_dim=1800)
        preview = draw_houses_on_preview(preview, houses, house_cord, preview_scale)
    else:
        log(f"Preview obrázok preskočený - veľká oblasť ({dem.shape[0]} x {dem.shape[1]} px). Uložia sa len CSV výstupy.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / "final_detector_output.png"
    if preview is not None and hasattr(preview, "size") and preview.size > 0:
        if preview is not None and hasattr(preview, "size") and preview.size > 0:
            cv2.imwrite(str(preview_path), preview)

    candidates_path = output_dir / "chunk_stats.csv"
    chunk_stats_df.to_csv(candidates_path, index=False)

    raw_csv_path = output_dir / "position_points.csv"
    wkt_csv_path = output_dir / "position_polygons.csv"

    points_df = houses_to_dataframe(houses, house_cord)
    points_df.to_csv(raw_csv_path, index=False)

    wkt_df = houses_to_wkt_dataframe(houses)
    wkt_df.to_csv(wkt_csv_path, index=False)

    summary = {
        "dem_shape": dem.shape,
        "used_chunks": use_chunks,
        "chunk_grid": (n_rows, n_cols) if use_chunks else None,
        "candidate_count": len(all_candidate_features),
        "house_segments": len(house),
        "final_count": len(houses),
        "class_counts": dict(counts),
        "preview_path": str(preview_path),
        "points_csv": str(raw_csv_path),
        "polygons_csv": str(wkt_csv_path),
        "chunk_stats_csv": str(candidates_path),
    }

    return summary


# -----------------------------
# Tkinter GUI
# -----------------------------
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DEM + mask detector GUI")
        self.root.geometry("1180x760")

        self.dem_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "gui_output"))
        self.chunk_mode = tk.StringVar(value="auto")
        self.chunk_rows = tk.IntVar(value=10)
        self.chunk_cols = tk.IntVar(value=10)

        self.preview_image = None

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # Inputs
        box = ttk.LabelFrame(left, text="Vstupy", padding=10)
        box.pack(fill="x", pady=(0, 10))

        self._file_row(box, "DEM (.tif/.npz/.npy):", self.dem_path, self.pick_dem)
        self._file_row(box, "Maska (.tif):", self.mask_path, self.pick_mask)
        self._file_row(box, "Output priečinok:", self.output_dir, self.pick_output_dir, select_dir=True)

        chunk_box = ttk.LabelFrame(left, text="Chunkovanie", padding=10)
        chunk_box.pack(fill="x", pady=(0, 10))

        ttk.Radiobutton(chunk_box, text="Automaticky podľa rozmerov", variable=self.chunk_mode, value="auto").pack(anchor="w")
        ttk.Radiobutton(chunk_box, text="Vždy zapnúť chunkovanie", variable=self.chunk_mode, value="yes").pack(anchor="w")
        ttk.Radiobutton(chunk_box, text="Nevyužiť chunkovanie", variable=self.chunk_mode, value="no").pack(anchor="w")

        grid_frame = ttk.Frame(chunk_box)
        grid_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(grid_frame, text="Riadky:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid_frame, from_=2, to=20, textvariable=self.chunk_rows, width=6).grid(row=0, column=1, padx=(5, 15))
        ttk.Label(grid_frame, text="Stĺpce:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(grid_frame, from_=2, to=20, textvariable=self.chunk_cols, width=6).grid(row=0, column=3, padx=(5, 0))

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_frame, text="Spustiť detekciu", command=self.run).pack(fill="x")

        summary_box = ttk.LabelFrame(left, text="Súhrn", padding=10)
        summary_box.pack(fill="both", expand=True)
        self.summary_text = tk.Text(summary_box, width=42, height=18, wrap="word")
        self.summary_text.pack(fill="both", expand=True)

        log_box = ttk.LabelFrame(right, text="Log", padding=10)
        log_box.pack(fill="x", pady=(0, 10))
        self.log_text = tk.Text(log_box, height=11, wrap="word")
        self.log_text.pack(fill="x")

        preview_box = ttk.LabelFrame(right, text="Preview výsledku", padding=10)
        preview_box.pack(fill="both", expand=True)

        preview_inner = ttk.Frame(preview_box)
        preview_inner.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(preview_inner)
        self.preview_label.pack(side="left", fill="both", expand=True)

        legend_box = ttk.LabelFrame(preview_inner, text="Legenda", padding=8)
        legend_box.pack(side="right", fill="y", padx=(10, 0))

        self.legend_canvas = tk.Canvas(legend_box, width=270, bg="white", highlightthickness=0)
        self.legend_canvas.pack(fill="both", expand=True)

    def _file_row(self, parent, label, variable, command, select_dir=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label).pack(anchor="w")
        entry_row = ttk.Frame(row)
        entry_row.pack(fill="x", pady=(2, 0))
        ttk.Entry(entry_row, textvariable=variable, width=45).pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Vybrať", command=command).pack(side="left", padx=(6, 0))

    def pick_dem(self):
        path = filedialog.askopenfilename(filetypes=[("DEM files", "*.tif *.tiff *.npz *.npy"), ("All files", "*.*")])
        if path:
            self.dem_path.set(path)

    def pick_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Mask TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.mask_path.set(path)

    def pick_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def set_summary(self, summary):
        self.summary_text.delete("1.0", "end")
        lines = [
            f"DEM shape: {summary['dem_shape']}",
            f"Chunkovanie: {'áno' if summary['used_chunks'] else 'nie'}",
        ]
        if summary['used_chunks']:
            lines.append(f"Grid: {summary['chunk_grid'][0]} x {summary['chunk_grid'][1]}")
        lines += [
            f"Candidate features: {summary['candidate_count']}",
            f"House segmenty: {summary['house_segments']}",
            f"Finálne objekty: {summary['final_count']}",
            "",
            "Rozdelenie tried:",
        ]

        if summary['class_counts']:
            for k, v in sorted(summary['class_counts'].items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- {k}: {v}")
        else:
            lines.append("- bez výsledkov")

        lines += [
            "",
            f"Preview PNG: {summary['preview_path']}",
            f"Points CSV: {summary['points_csv']}",
            f"Polygons CSV: {summary['polygons_csv']}",
            f"Chunk stats CSV: {summary['chunk_stats_csv']}",
        ]

        self.summary_text.insert("1.0", "\n".join(lines))

    def show_preview(self, image_path, class_counts=None):
        img = Image.open(image_path)
        max_w, max_h = 760, 520
        img.thumbnail((max_w, max_h))
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_image)

        if hasattr(self, "legend_canvas"):
            self.legend_canvas.delete("all")
            y = 20
            self.legend_canvas.create_text(12, y, anchor="w", text="Legenda", font=("TkDefaultFont", 10, "bold"))
            y += 26

            if class_counts:
                for label, count in sorted(class_counts.items(), key=lambda x: (-x[1], x[0])):
                    b, g, r = COLOR_MAP.get(label, (255, 255, 255))
                    color_hex = f"#{r:02x}{g:02x}{b:02x}"
                    self.legend_canvas.create_oval(12, y - 2, 28, y + 14, fill=color_hex, outline="#444444")
                    self.legend_canvas.create_text(40, y + 6, anchor="w", text=DISPLAY_LABELS.get(label, label))
                    y += 26

    def validate_inputs(self):
        if not self.dem_path.get().strip():
            raise ValueError("Chýba DEM súbor.")
        if not self.mask_path.get().strip():
            raise ValueError("Chýba maska.")
        if not os.path.exists(self.dem_path.get()):
            raise ValueError("DEM súbor neexistuje.")
        if not os.path.exists(self.mask_path.get()):
            raise ValueError("Maska neexistuje.")
        if self.chunk_mode.get() == "yes":
            if self.chunk_rows.get() < 2 or self.chunk_cols.get() < 2:
                raise ValueError("Pri nútenom chunkovaní musia byť riadky aj stĺpce aspoň 2.")

    def run(self):
        self.log_text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")
        self.preview_label.configure(image="")
        self.preview_image = None
        if hasattr(self, "legend_canvas"):
            self.legend_canvas.delete("all")

        try:
            self.validate_inputs()
            summary = run_pipeline(
                dem_path=self.dem_path.get(),
                mask_path=self.mask_path.get(),
                output_dir=self.output_dir.get(),
                chunk_mode=self.chunk_mode.get(),
                force_rows=self.chunk_rows.get(),
                force_cols=self.chunk_cols.get(),
                logger=self.log,
            )
            self.set_summary(summary)
            self.show_preview(summary['preview_path'], summary.get('class_counts'))
            messagebox.showinfo("Hotovo", "Detekcia bola úspešne dokončená.")
        except Exception as e:
            self.log("CHYBA:")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("Chyba", str(e))


def main():
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
