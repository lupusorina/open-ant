"""Browser-based 3D viewer for framing snapshots (EGL, no display needed).

    ant_env/bin/python sim/view.py                   # ant
    ant_env/bin/python sim/view.py --scene crane     # electric crane
    (also accepts snapshot.py's options, e.g. --light_floor, --fovy, --port 8001)

Then open http://localhost:8000 in your browser. Over SSH, forward the port first:
    ssh -L 8000:localhost:8000 <this machine>
(VS Code Remote forwards it automatically.)

The server renders every frame with EGL on the GPU and streams it as a JPEG, so the preview
is exactly what the final snapshot will look like (same renderer, same 16:9 framing).

Controls in the page:
    left-drag      rotate           right-drag / shift-drag   pan
    scroll         zoom             sliders                   pose (per scene), field of view
    Enter or "Save 4K"  -> writes <out_dir>/view_<scene>_<time>.json and renders the .png at full
                           quality in the background (same as snapshot.py --view <json>).
"""

import io
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

os.environ["MUJOCO_GL"] = "egl"

import mujoco
from PIL import Image

from scenes import HERE, apply_quality
from snapshot import default_camera, make_parser, make_renderer, scene_option

PREVIEW_W, PREVIEW_H = 1280, 720

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__ Viewer</title>
<style>
  :root { --bg:#111; --panel:#1c1c1c; --fg:#ddd; --muted:#888; --accent:#4a9eff; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px system-ui,sans-serif;
         display:flex; gap:12px; padding:12px; box-sizing:border-box; flex-wrap:wrap; }
  #view { width:min(100%, 1280px); aspect-ratio:16/9; background:#000; cursor:grab;
          user-select:none; -webkit-user-drag:none; display:block; }
  #panel { background:var(--panel); padding:12px; border-radius:6px; width:260px; }
  label { display:flex; justify-content:space-between; color:var(--muted); margin-top:6px; }
  input[type=range] { width:100%; }
  button { margin-top:10px; width:100%; padding:8px; background:var(--accent); color:#fff;
           border:0; border-radius:4px; font-weight:600; cursor:pointer; }
  button.sec { background:#333; }
  #cam, #status { font-family:ui-monospace,monospace; font-size:12px; color:var(--muted);
                  white-space:pre; margin-top:10px; }
</style></head><body>
<img id="view" draggable="false">
<div id="panel">
  <b>Pose</b><div id="pose"></div>
  <b style="display:block;margin-top:12px">Lens</b>
  <label>fovy <span id="fovyv"></span></label><input type="range" id="fovy" min="10" max="90" step="1">
  <button id="save">Save 4K (Enter)</button>
  <button class="sec" id="reset">Reset pose + camera</button>
  <div id="cam"></div><div id="status"></div>
</div>
<script>
const SLIDERS = __SLIDERS__, CAM0 = __CAM0__;
const POSE0 = SLIDERS.map(s => s.value);
let cam = structuredClone(CAM0), pose = POSE0.slice(), fovy = __FOVY__;
const img = document.getElementById('view');
let busy = false, dirty = true, poseDirty = false;

const poseDiv = document.getElementById('pose');
SLIDERS.forEach((s, i) => {
  poseDiv.insertAdjacentHTML('beforeend',
    `<label>${s.name} <span id="v${i}"></span></label>
     <input type="range" id="s${i}" min="${s.min}" max="${s.max}" step="${(s.max - s.min) / 400}">`);
});
function syncUI() {
  SLIDERS.forEach((s, i) => { document.getElementById('s'+i).value = pose[i];
                              document.getElementById('v'+i).textContent = pose[i].toFixed(2); });
  document.getElementById('fovy').value = fovy;
  document.getElementById('fovyv').textContent = fovy;
  document.getElementById('cam').textContent =
    `azimuth   ${cam.azimuth.toFixed(1)}\nelevation ${cam.elevation.toFixed(1)}\n` +
    `distance  ${cam.distance.toFixed(3)}\nlookat    ${cam.lookat.map(v=>v.toFixed(3)).join(' ')}`;
}
SLIDERS.forEach((s, i) => document.getElementById('s'+i).addEventListener('input', e => {
  pose[i] = +e.target.value; poseDirty = dirty = true; syncUI(); }));
document.getElementById('fovy').addEventListener('input', e => { fovy = +e.target.value; dirty = true; syncUI(); });

function frame() {
  if (busy || !dirty) return;
  busy = true; dirty = false;
  const q = new URLSearchParams({ az: cam.azimuth, el: cam.elevation, dist: cam.distance,
    lookat: cam.lookat.join(','), fovy, t: Date.now() });
  if (poseDirty) { q.set('pose', pose.join(',')); poseDirty = false; }
  const next = new Image();
  next.onload = () => { img.src = next.src; busy = false; };
  next.onerror = () => { busy = false; };
  next.src = '/frame?' + q;
}
setInterval(frame, 16);

// mouse: MuJoCo free-camera convention (forward = cos(el)cos(az), cos(el)sin(az), sin(el))
let drag = null;
img.addEventListener('contextmenu', e => e.preventDefault());
img.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY,
  pan: e.button === 2 || e.shiftKey }; img.style.cursor = 'grabbing'; });
window.addEventListener('mouseup', () => { drag = null; img.style.cursor = 'grab'; });
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const az = cam.azimuth * Math.PI/180, el = cam.elevation * Math.PI/180;
    const right = [Math.sin(az), -Math.cos(az), 0];
    const fwd = [Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el)];
    const up = [right[1]*fwd[2]-right[2]*fwd[1], right[2]*fwd[0]-right[0]*fwd[2], right[0]*fwd[1]-right[1]*fwd[0]];
    const s = cam.distance * 2 * Math.tan(fovy*Math.PI/360) / img.clientHeight;
    for (let k = 0; k < 3; k++) cam.lookat[k] += s * (-dx*right[k] + dy*up[k]);
  } else {
    cam.azimuth = (cam.azimuth - dx*0.3) % 360;
    cam.elevation = Math.max(-89, Math.min(89, cam.elevation - dy*0.3));
  }
  dirty = true; syncUI();
});
img.addEventListener('wheel', e => { e.preventDefault();
  cam.distance = Math.max(0.05, cam.distance * Math.exp(e.deltaY * 0.001)); dirty = true; syncUI();
}, { passive: false });

async function save() {
  const st = document.getElementById('status');
  st.textContent = 'saving...';
  const r = await fetch('/save', { method: 'POST', body: JSON.stringify({ cam, fovy }) });
  st.textContent = (await r.json()).msg;
}
document.getElementById('save').onclick = save;
window.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
document.getElementById('reset').onclick = () => {
  cam = structuredClone(CAM0); pose = POSE0.slice(); poseDirty = dirty = true; syncUI(); };
syncUI();
</script></body></html>
"""


def main():
    p, scene = make_parser(lambda p: p.add_argument("--port", type=int, default=8000))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    passthrough = list(sys.argv[1:])
    if "--port" in passthrough:  # snapshot.py doesn't know --port
        i = passthrough.index("--port")
        del passthrough[i:i + 2]

    model = scene.build(args)
    apply_quality(model, scene, args, offscreen_size=(PREVIEW_W, PREVIEW_H))
    data = mujoco.MjData(model)
    scene.reset(model, data, args)
    cam = default_camera(scene, model, data, args)

    # One renderer, used only from the (single-threaded) server loop -> EGL context stays on one thread.
    renderer = make_renderer(model, PREVIEW_W, PREVIEW_H)
    opt = scene_option(scene)

    page = (PAGE
            .replace("__TITLE__", args.scene.capitalize())
            .replace("__SLIDERS__", json.dumps(scene.sliders(model, data)))
            .replace("__FOVY__", json.dumps(args.fovy))
            .replace("__CAM0__", json.dumps({
                "azimuth": cam.azimuth, "elevation": cam.elevation,
                "distance": cam.distance, "lookat": cam.lookat.tolist()})))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self._send(page.encode(), "text/html; charset=utf-8")
            if url.path != "/frame":
                return self._send(b"not found", "text/plain", 404)

            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if "pose" in q:
                scene.apply(model, data, [float(v) for v in q["pose"].split(",")])
            cam.azimuth = float(q["az"])
            cam.elevation = float(q["el"])
            cam.distance = float(q["dist"])
            cam.lookat[:] = [float(v) for v in q["lookat"].split(",")]
            model.vis.global_.fovy = float(q["fovy"])

            renderer.update_scene(data, camera=cam, scene_option=opt)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, "JPEG", quality=90)
            self._send(buf.getvalue(), "image/jpeg")

        def do_POST(self):
            if self.path != "/save":
                return self._send(b"not found", "text/plain", 404)
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            c = req["cam"]
            view = {
                "scene": args.scene,
                "qpos": data.qpos.tolist(),
                "ctrl": data.ctrl.tolist(),
                "act": data.act.tolist(),
                "lookat": c["lookat"],
                "distance": c["distance"],
                "azimuth": c["azimuth"],
                "elevation": c["elevation"],
                "fovy": req["fovy"],
            }
            path = os.path.join(args.out_dir, time.strftime(f"view_{args.scene}_%Y%m%d-%H%M%S.json"))
            with open(path, "w") as f:
                json.dump(view, f, indent=2)
            # Full-quality render in a separate process (big offscreen buffer, supersampling).
            subprocess.Popen([sys.executable, os.path.join(HERE, "snapshot.py"),
                              *passthrough, "--view", path])
            png = path[:-5] + ".png"
            print(f"saved {path} -> rendering {png}")
            self._send(json.dumps({"msg": f"rendering\n{os.path.basename(png)}"}).encode(),
                       "application/json")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{args.scene} viewer running at http://localhost:{args.port}  (Ctrl+C to stop)")
    print(f"Over SSH: ssh -L {args.port}:localhost:{args.port} <this machine>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
