import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";

interface Point {
  x: number;
  y: number;
}

interface Zone {
  id: string;
  name: string;
  color: string;
  points: Point[];
}

const CLOSE_RADIUS = 10;

const PALETTE = [
  "#ef4444",
  "#f97316",
  "#eab308",
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#ec4899",
];

function dist(a: Point, b: Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function drawZone(ctx: CanvasRenderingContext2D, zone: Zone) {
  if (zone.points.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(zone.points[0].x, zone.points[0].y);

  zone.points.slice(1).forEach((p) => {
    ctx.lineTo(p.x, p.y);
  });

  ctx.closePath();

  ctx.fillStyle = hexToRgba(zone.color, 0.25);
  ctx.fill();

  ctx.strokeStyle = zone.color;
  ctx.lineWidth = 2;
  ctx.stroke();

  const cx =
    zone.points.reduce((s, p) => s + p.x, 0) / zone.points.length;

  const cy =
    zone.points.reduce((s, p) => s + p.y, 0) / zone.points.length;

  ctx.font = "bold 13px monospace";
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  ctx.fillText(zone.name, cx, cy);
}

function drawDraft(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  mouse: Point | null,
  color: string
) {
  if (points.length === 0) return;

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);

  points.slice(1).forEach((p) => {
    ctx.lineTo(p.x, p.y);
  });

  if (mouse) {
    ctx.lineTo(mouse.x, mouse.y);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  points.forEach((p, i) => {
    ctx.beginPath();

    ctx.arc(
      p.x,
      p.y,
      i === 0 ? CLOSE_RADIUS : 5,
      0,
      Math.PI * 2
    );

    ctx.fillStyle =
      i === 0 ? hexToRgba(color, 0.4) : color;

    ctx.fill();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

export default function ZoneEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const [zones, setZones] = useState<Zone[]>([]);
  const [draft, setDraft] = useState<Point[]>([]);
  const [mouse, setMouse] = useState<Point | null>(null);

  const [zoneName, setZoneName] = useState("");
  const [zoneColor, setZoneColor] = useState(PALETTE[0]);

  const [nameError, setNameError] = useState("");

  const [status, setStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");

  const [snapshotError, setSnapshotError] =
    useState(false);

  useEffect(() => {
    const img = new Image();

    img.onload = () => {
      imgRef.current = img;
      renderCanvas();
    };

    img.onerror = () => {
      setSnapshotError(true);
    };

    img.src = "/api/snapshot";
  }, []);

  useEffect(() => {
    fetch("/zones")
      .then((r) => r.json())
      .then((data: Zone[]) => {
        setZones(data);
      })
      .catch(() => {});
  }, []);

  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imgRef.current) {
      ctx.drawImage(
        imgRef.current,
        0,
        0,
        canvas.width,
        canvas.height
      );
    } else {
      ctx.fillStyle = "#111827";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "#6b7280";
      ctx.font = "16px monospace";

      ctx.fillText(
        snapshotError
          ? "Snapshot unavailable"
          : "Loading snapshot...",
        30,
        40
      );
    }

    zones.forEach((z) => drawZone(ctx, z));

    drawDraft(ctx, draft, mouse, zoneColor);
  }, [zones, draft, mouse, zoneColor, snapshotError]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  function getCanvasPoint(
    e: React.MouseEvent<HTMLCanvasElement>
  ): Point {
    const rect =
      canvasRef.current!.getBoundingClientRect();

    const scaleX =
      canvasRef.current!.width / rect.width;

    const scaleY =
      canvasRef.current!.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  function handleCanvasClick(
    e: React.MouseEvent<HTMLCanvasElement>
  ) {
    const pt = getCanvasPoint(e);

    if (
      draft.length >= 3 &&
      dist(pt, draft[0]) <= CLOSE_RADIUS
    ) {
      closePolygon();
      return;
    }

    setDraft((prev) => [...prev, pt]);
  }

  function handleMouseMove(
    e: React.MouseEvent<HTMLCanvasElement>
  ) {
    setMouse(getCanvasPoint(e));
  }

  function closePolygon() {
    if (draft.length < 3) return;

    const trimmed = zoneName.trim();

    if (!trimmed) {
      setNameError("Zone name is required.");
      return;
    }

    if (
      zones.some(
        (z) =>
          z.name.toLowerCase() ===
          trimmed.toLowerCase()
      )
    ) {
      setNameError("Zone name must be unique.");
      return;
    }

    const newZone: Zone = {
      id: crypto.randomUUID(),
      name: trimmed,
      color: zoneColor,
      points: draft,
    };

    setZones((prev) => [...prev, newZone]);

    setDraft([]);
    setMouse(null);

    setZoneName("");
    setNameError("");

    setZoneColor(PALETTE[0]);
  }

  function cancelDraft() {
    setDraft([]);
    setMouse(null);
    setNameError("");
  }

  function deleteZone(id: string) {
    setZones((prev) =>
      prev.filter((z) => z.id !== id)
    );
  }

  async function saveZones() {
    setStatus("saving");

    try {
      const res = await fetch("/zones", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(zones),
      });

      if (!res.ok) {
        throw new Error();
      }

      setStatus("saved");

      setTimeout(() => {
        setStatus("idle");
      }, 2000);
    } catch {
      setStatus("error");

      setTimeout(() => {
        setStatus("idle");
      }, 3000);
    }
  }

  const cursorStyle =
    draft.length >= 3 &&
    mouse &&
    dist(mouse, draft[0]) <= CLOSE_RADIUS
      ? "cell"
      : "crosshair";

  return (
    <div style={styles.root}>
      <h2 style={styles.heading}>
        Restricted Zone Editor
      </h2>

      <div style={styles.canvasWrapper}>
        <canvas
          ref={canvasRef}
          width={960}
          height={540}
          style={{
            ...styles.canvas,
            cursor: cursorStyle,
          }}
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setMouse(null)}
        />

        {draft.length > 0 && (
          <div style={styles.badge}>
            Drawing... {draft.length} point
            {draft.length !== 1 ? "s" : ""}
          </div>
        )}
      </div>

      <div style={styles.controls}>
        <div style={styles.field}>
          <label style={styles.label}>
            Zone Name
          </label>

          <input
            value={zoneName}
            onChange={(e) => {
              setZoneName(e.target.value);
              setNameError("");
            }}
            placeholder="restricted_door_zone"
            style={{
              ...styles.input,
              borderColor: nameError
                ? "#ef4444"
                : "#334155",
            }}
          />

          {nameError && (
            <span style={styles.error}>
              {nameError}
            </span>
          )}
        </div>

        <div style={styles.field}>
          <label style={styles.label}>
            Color
          </label>

          <div style={styles.palette}>
            {PALETTE.map((c) => (
              <button
                key={c}
                style={{
                  ...styles.swatch,
                  backgroundColor: c,
                }}
                onClick={() => setZoneColor(c)}
              />
            ))}
          </div>
        </div>

        <div style={styles.actions}>
          {draft.length >= 3 && (
            <button
              style={styles.primaryBtn}
              onClick={closePolygon}
            >
              Close Polygon
            </button>
          )}

          {draft.length > 0 && (
            <button
              style={styles.ghostBtn}
              onClick={cancelDraft}
            >
              Cancel
            </button>
          )}

          {zones.length > 0 && (
            <button
              style={styles.saveBtn}
              onClick={saveZones}
            >
              Save Zones
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    padding: "24px",
    background: "#0f172a",
    minHeight: "100vh",
    color: "#ffffff",
    fontFamily: "sans-serif",
  },

  heading: {
    marginBottom: "16px",
  },

  canvasWrapper: {
    position: "relative",
    display: "inline-block",
  },

  canvas: {
    border: "1px solid #334155",
    borderRadius: "8px",
    maxWidth: "100%",
  },

  badge: {
    position: "absolute",
    bottom: "10px",
    left: "10px",
    background: "#111827",
    padding: "6px 10px",
    borderRadius: "6px",
    fontSize: "12px",
  },

  controls: {
    marginTop: "20px",
    display: "flex",
    gap: "20px",
    flexWrap: "wrap",
    alignItems: "flex-end",
  },

  field: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },

  label: {
    fontSize: "13px",
  },

  input: {
    background: "#1e293b",
    color: "#ffffff",
    border: "1px solid #334155",
    borderRadius: "6px",
    padding: "10px",
    width: "240px",
    outline: "none",
  },

  error: {
    color: "#ef4444",
    fontSize: "12px",
  },

  palette: {
    display: "flex",
    gap: "8px",
  },

  swatch: {
    width: "24px",
    height: "24px",
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
  },

  actions: {
    display: "flex",
    gap: "10px",
  },

  primaryBtn: {
    background: "#0ea5e9",
    border: "none",
    color: "#ffffff",
    padding: "10px 16px",
    borderRadius: "6px",
    cursor: "pointer",
  },

  ghostBtn: {
    background: "transparent",
    border: "1px solid #334155",
    color: "#ffffff",
    padding: "10px 16px",
    borderRadius: "6px",
    cursor: "pointer",
  },

  saveBtn: {
    background: "#22c55e",
    border: "none",
    color: "#ffffff",
    padding: "10px 16px",
    borderRadius: "6px",
    cursor: "pointer",
  },
};