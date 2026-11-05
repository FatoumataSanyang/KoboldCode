#!/usr/bin/env python3
"""
KoboldCode - a Claude-style chat app + a Claude-Code-style coding agent for local models.

Talks to KoboldCpp through BOTH of its APIs (OpenAI-compatible chat and the native
KoboldAI generate stream). Standard library only - nothing to pip install.

    python kobold_code.py                          # opens http://127.0.0.1:8765
    python kobold_code.py --kobold http://localhost:5001
    python kobold_code.py --window                 # native window (pip install pywebview)

Start KoboldCpp first, e.g.:  koboldcpp --model your-model.gguf --port 5001 --contextsize 16384
All data lives in ~/.koboldcode/state.json. Agent projects default to ~/KoboldCode/.
"""
import argparse, atexit, base64, io, json, os, platform, re, secrets, shutil, signal, socket, struct, subprocess, sys, threading, time, uuid
import urllib.error, urllib.parse, urllib.request, webbrowser, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path.home() / ".koboldcode"
APP.mkdir(exist_ok=True)
STATE = APP / "state.json"
DEFAULT_WS = Path.home() / "KoboldCode"
TOKEN = secrets.token_urlsafe(18)          # blocks other websites from driving the local server
IGNORE = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".idea", ".kobold"}
PY = "python" if os.name == "nt" or not shutil.which("python3") else "python3"
CFG = {"kobold": "http://localhost:5001", "host": "127.0.0.1"}
UNDO = {}

# --------------------------------------------------------------------------- workspace tools
def safe(ws, rel="."):
    root = Path(ws).expanduser().resolve()
    p = (root / (rel or ".")).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError(f"path '{rel}' is outside the workspace")
    return root, p


def iter_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in IGNORE)
        for f in sorted(fn):
            yield Path(dp) / f


def rel_of(root, p):
    return p.relative_to(root).as_posix() or "."


def push_undo(root, p):
    prev = p.read_bytes() if p.is_file() else None
    st = UNDO.setdefault(str(root), [])
    st.append((str(p), prev))
    del st[:-200]


def undo(ws):
    st = UNDO.get(str(Path(ws).expanduser().resolve()))
    if not st:
        return {"ok": False, "out": "Nothing to undo."}
    p, prev = st.pop()
    p = Path(p)
    if prev is None:
        if p.exists():
            p.unlink()
        return {"ok": True, "out": f"Removed {p.name}"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(prev)
    return {"ok": True, "out": f"Restored {p.name}"}


def tree_text(root, depth=2, limit=150):
    out = []

    def walk(d, lvl):
        try:
            items = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except OSError:
            return
        for it in items:
            if it.name in IGNORE or len(out) >= limit:
                continue
            out.append("  " * lvl + it.name + ("/" if it.is_dir() else ""))
            if it.is_dir() and lvl + 1 < depth:
                walk(it, lvl + 1)

    walk(root, 0)
    return "\n".join(out) or "(empty)"


def t_list(ws, a):
    root, p = safe(ws, a.get("path") or ".")
    if not p.is_dir():
        return False, f"not a directory: {a.get('path')}"
    return True, tree_text(p, int(a.get("depth") or 1), 300)


def t_read(ws, a):
    root, p = safe(ws, a.get("path"))
    if not p.is_file():
        return False, f"file not found: {a.get('path')}"
    if p.stat().st_size > 1_500_000:
        return False, "file too large to read"
    try:
        lines = p.read_text("utf-8").split("\n")
    except UnicodeDecodeError:
        return False, "binary file"
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    s = max(int(a.get("start") or 1), 1)
    e = min(int(a.get("end") or s + 399), s + 599, len(lines))
    body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(s, e + 1)) or "(empty file)"
    note = f"\n[showing lines {s}-{e} of {len(lines)}]" if (s > 1 or e < len(lines)) else ""
    return True, body + note


def t_write(ws, a):
    root, p = safe(ws, a.get("path"))
    if p == root or p.is_dir():
        return False, "path is a directory"
    c = a.get("content") or ""
    if c and not c.endswith("\n"):
        c += "\n"
    push_undo(root, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(c)
    return True, f"Wrote {rel_of(root, p)} ({c.count(chr(10))} lines, {len(c.encode())} bytes)"


def lenient(text, old, new):
    """Match ignoring indentation/trailing-space differences - small local models often drift there."""
    tl, ol = text.split("\n"), [x.strip() for x in old.strip("\n").split("\n")]
    for i in range(len(tl) - len(ol) + 1):
        if [x.strip() for x in tl[i:i + len(ol)]] == ol:
            return "\n".join(tl[:i] + new.strip("\n").split("\n") + tl[i + len(ol):])
    return None


def t_edit(ws, a):
    root, p = safe(ws, a.get("path"))
    if not p.is_file():
        return False, "file not found (use write_file to create it)"
    old, new = a.get("old") or "", a.get("new") or ""
    if not old.strip():
        return False, "<old> is empty"
    text = p.read_text("utf-8")
    n = text.count(old)
    if n > 1:
        return False, f"<old> matches {n} places; include more surrounding lines so it is unique"
    res = text.replace(old, new) if n == 1 else lenient(text, old, new)
    if res is None:
        return False, "<old> text not found. read_file again and copy the exact lines (without the line-number prefix)."
    push_undo(root, p)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(res)
    return True, f"Edited {rel_of(root, p)} (-{old.count(chr(10)) + 1} +{new.count(chr(10)) + 1} lines)"


def t_delete(ws, a):
    root, p = safe(ws, a.get("path"))
    if p == root:
        return False, "refusing to delete the workspace root"
    if p.is_file():
        push_undo(root, p)
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    else:
        return False, "no such path"
    return True, f"Deleted {rel_of(root, p)}"


def t_search(ws, a):
    root, p = safe(ws, a.get("path") or ".")
    pat = a.get("pattern") or ""
    try:
        rx = re.compile(pat, re.I)
    except re.error:
        rx = re.compile(re.escape(pat), re.I)
    hits = []
    for f in ([p] if p.is_file() else iter_files(p)):
        try:
            if f.stat().st_size > 1_000_000:
                continue
            for i, line in enumerate(f.read_text("utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel_of(root, f)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 80:
                        return True, "\n".join(hits) + "\n[stopped at 80 matches]"
        except (UnicodeDecodeError, OSError):
            continue
    return True, "\n".join(hits) or "no matches"


def kill_tree(pr):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pr.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
    except Exception:
        pr.kill()


def t_run(ws, a):
    cmd = (a.get("command") or "").strip()
    if not cmd:
        return False, "empty command"
    root, _ = safe(ws)
    timeout = min(int(a.get("timeout") or 60), 300)
    kw = dict(cwd=root, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
              text=True, encoding="utf-8", errors="replace",
              env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "CI": "1"})
    if os.name != "nt":
        kw["start_new_session"] = True
    pr = subprocess.Popen(cmd, **kw)
    try:
        out, _ = pr.communicate(timeout=timeout)
        code = pr.returncode
    except subprocess.TimeoutExpired:
        kill_tree(pr)
        try:
            out, _ = pr.communicate(timeout=3)
        except Exception:
            out = ""
        code = f"timeout after {timeout}s"
    out = out or ""
    if len(out) > 6000:
        out = "...[truncated]...\n" + out[-6000:]
    return code == 0, f"{out.rstrip()}\n[exit: {code}]".lstrip()


TOOLS = {"list_dir": t_list, "read_file": t_read, "write_file": t_write, "edit_file": t_edit,
         "delete_path": t_delete, "search": t_search, "run_command": t_run}


def run_tool(ws, name, args):
    fn = TOOLS.get(name)
    if not fn:
        return {"ok": False, "out": f"unknown tool '{name}'. Available: {', '.join(TOOLS)}, todo, finish"}
    try:
        ok, out = fn(ws, args or {})
    except Exception as e:
        ok, out = False, f"{type(e).__name__}: {e}"
    return {"ok": ok, "out": out}


def ws_context(path):
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    mem = root / "KOBOLD.md"
    return {"cwd": str(root), "tree": tree_text(root), "os": f"{platform.system()} {platform.release()}",
            "py": PY, "memory": mem.read_text("utf-8", "replace")[:4000] if mem.is_file() else ""}


def ws_tree(path):
    root = Path(path).expanduser().resolve()
    return [{"p": rel_of(root, f), "s": f.stat().st_size} for _, f in zip(range(1500), iter_files(root))] if root.is_dir() else []


def make_zip(root):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in iter_files(root):
            z.write(f, f.relative_to(root).as_posix())
    return buf.getvalue()


def pick_dir():
    try:
        if sys.platform == "darwin":  # Tk must run on the main thread on macOS
            r = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder)'], capture_output=True, text=True)
            return r.stdout.strip()
        import tkinter
        from tkinter import filedialog
        t = tkinter.Tk()
        t.withdraw()
        t.attributes("-topmost", True)
        d = filedialog.askdirectory()
        t.destroy()
        return d or ""
    except Exception:
        return ""


def open_folder(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# --------------------------------------------------------------------------- KoboldCpp
def kreq(base, path, key="", body=None, timeout=5):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + path, data, h), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "{}")


def test_conn(url, key):
    out = {"ok": False}
    try:
        out["model"] = kreq(url, "/api/v1/model", key).get("result", "")
        out["ok"] = True
    except Exception:
        try:
            out["model"] = ((kreq(url, "/v1/models", key).get("data") or [{}])[0]).get("id", "")
            out["ok"] = True
        except Exception as e:
            out["err"] = f"Cannot reach {url} ({e}). Is KoboldCpp running?"
            return out
    for k, path, f in (("ctx", "/api/extra/true_max_context_length", lambda j: j.get("value")),
                       ("version", "/api/extra/version", lambda j: j.get("version"))):
        try:
            out[k] = f(kreq(url, path, key, timeout=3))
        except Exception:
            pass
    return out



# --------------------------------------------------------------------------- Discord Rich Presence
class IPC:
    """One connection to the local Discord client (unix socket, or a named pipe on Windows)."""
    def __init__(self, path):
        self.f = self.s = None
        if os.name == "nt":
            self.f = open(path, "r+b", buffering=0)
        else:
            self.s = socket.socket(socket.AF_UNIX)
            self.s.settimeout(8)
            self.s.connect(path)

    def write(self, b):
        self.f.write(b) if self.f else self.s.sendall(b)

    def read(self, n):
        buf = b""
        while len(buf) < n:
            c = self.f.read(n - len(buf)) if self.f else self.s.recv(n - len(buf))
            if not c:
                raise OSError("Discord closed the connection")
            buf += c
        return buf

    def close(self):
        try:
            (self.f or self.s).close()
        except Exception:
            pass


class Discord:
    """Minimal Rich Presence client. You need your own (free) Discord application ID - see Customize > Discord."""
    def __init__(self):
        self.cid, self.enabled, self.act = "", False, {}
        self.conn, self.status, self.msg, self.user = None, "off", "", ""
        self.sent, self.retry, self.dirty = 0.0, 0.0, False
        self.start = int(time.time())
        threading.Thread(target=self.loop, daemon=True).start()

    @staticmethod
    def paths():
        if os.name == "nt":
            return [r"\\?\pipe\discord-ipc-%d" % i for i in range(10)]
        out = []
        for base in [os.environ.get(k) for k in ("XDG_RUNTIME_DIR", "TMPDIR", "TMP", "TEMP")] + ["/tmp"]:
            for i in range(10):
                for sub in ("", "snap.discord/", "app/com.discordapp.Discord/"):
                    if base:
                        out.append(f"{base}/{sub}discord-ipc-{i}")
        return out

    def update(self, b):
        cid = str(b.get("client_id") or "").strip()
        act = {k: str(b.get(k) or "")[:120] for k in ("details", "state", "large_image", "large_text")}
        en = bool(b.get("enabled")) and bool(re.fullmatch(r"\d{15,25}", cid))
        if cid != self.cid:
            self.close()
        if (act, en) != (self.act, self.enabled):
            self.dirty = True
        self.cid, self.enabled, self.act = cid, en, act
        if b.get("enabled") and not en:
            self.status, self.msg = "error", "The Application ID should be a 17-19 digit number."
        return self.info()

    def info(self):
        return {"status": self.status, "msg": self.msg, "user": self.user}

    def send(self, op, payload):
        data = json.dumps(payload).encode()
        self.conn.write(struct.pack("<II", op, len(data)) + data)

    def recv(self):
        op, n = struct.unpack("<II", self.conn.read(8))
        return op, json.loads(self.conn.read(n) or b"{}")

    def close(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        self.user = ""

    def connect(self):
        for p in self.paths():
            try:
                self.conn = IPC(p)
                break
            except OSError:
                continue
        if not self.conn:
            raise OSError("Discord is not running (no IPC pipe found)")
        self.send(0, {"v": 1, "client_id": self.cid})
        _, d = self.recv()
        if d.get("evt") != "READY":
            raise OSError("Discord refused the Application ID: " + str((d.get("data") or {}).get("message", d))[:100])
        self.user, self.status, self.msg = (d.get("data") or {}).get("user", {}).get("username", ""), "connected", ""

    def send_act(self, act):
        a = None
        if act:
            a = {"details": act.get("details"), "state": act.get("state"), "timestamps": {"start": self.start}}
            if act.get("large_image"):
                a["assets"] = {"large_image": act["large_image"], "large_text": act.get("large_text") or "KoboldCode"}
            a = {k: v for k, v in a.items() if v and (not isinstance(v, str) or len(v) >= 2)}
        self.send(1, {"cmd": "SET_ACTIVITY", "args": {"pid": os.getpid(), "activity": a}, "nonce": uuid.uuid4().hex})
        _, d = self.recv()
        if d.get("evt") == "ERROR":
            raise OSError((d.get("data") or {}).get("message", "SET_ACTIVITY failed"))

    def loop(self):
        while True:
            time.sleep(1)
            try:
                if not self.enabled:
                    if self.conn:
                        self.send_act(None)
                        self.close()
                    if self.status != "error":
                        self.status, self.msg = "off", ""
                    continue
                if self.conn is None:
                    if time.time() < self.retry:
                        continue
                    self.connect()
                    self.dirty = True
                if self.dirty and time.time() - self.sent >= 15:   # Discord rate-limits presence updates
                    self.send_act(self.act)
                    self.dirty, self.sent = False, time.time()
            except Exception as e:
                self.close()
                self.status, self.msg, self.retry = "waiting", str(e)[:140], time.time() + 15


DISCORD = Discord()
atexit.register(lambda: DISCORD.conn and DISCORD.send_act(None))
WALL = APP / "wallpaper.jpg"

# --------------------------------------------------------------------------- HTTP
UI = r'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KoboldCode</title>
<style>
:root{--bg:#262624;--side:#1f1e1d;--card:#30302e;--card2:#3a3a37;--bd:#3f3e3a;--tx:#faf9f5;--mu:#a3a199;--ac:#d97757;--ok:#86b97e;--er:#e5726a;--code:#1c1c1b;--hov:#ffffff12;
--serif:"Tiempos Text","Source Serif 4","Iowan Old Style","Palatino Linotype",Georgia,serif;
--sans:-apple-system,"Segoe UI",Inter,Roboto,"Helvetica Neue",sans-serif;
--mono:"JetBrains Mono","Cascadia Code","SF Mono",Consolas,Menlo,monospace;--ft:var(--serif)}
[data-theme=light]{--bg:#faf9f5;--side:#f0eee6;--card:#fff;--card2:#f3f1e8;--bd:#e2dfd2;--tx:#1b1b19;--mu:#77756c;--code:#f5f3ea;--hov:#00000009}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 var(--sans);overflow:hidden;-webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer;padding:0;text-align:left}
a{color:var(--ac)}
input,textarea,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;outline:none;width:100%}
input:focus,textarea:focus,select:focus{border-color:var(--ac)}
input[type=range]{padding:0;accent-color:var(--ac)}
kbd{font:11px var(--sans);color:var(--mu);margin-left:auto;border:1px solid var(--bd);border-radius:5px;padding:0 5px}
::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:var(--bd);border-radius:9px}
svg.i{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
#app{display:flex;height:100vh}
.sp{flex:1}.mu{color:var(--mu)}

/* sidebar */
.side{width:272px;flex:none;background:var(--side);border-right:1px solid var(--bd);display:flex;flex-direction:column;padding:12px 10px}
.brand{display:flex;align-items:center;gap:9px;padding:4px 8px 12px;font:600 17px var(--serif)}
.seg{display:flex;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:3px;margin:0 2px 10px}
.seg button{flex:1;display:flex;gap:6px;align-items:center;justify-content:center;padding:6px;border-radius:7px;color:var(--mu)}
.seg button.on{background:var(--card2);color:var(--tx)}
.nav{display:flex;align-items:center;gap:10px;width:100%;padding:7px 9px;border-radius:8px}
.nav:hover,.li:hover{background:var(--hov)}.nav.on{background:var(--hov)}
.nav:first-child{color:var(--ac);font-weight:600}
.list{flex:1;overflow:auto;margin-top:10px}
.sh{font-size:12px;color:var(--mu);padding:10px 9px 4px}
.li{display:flex;align-items:center;gap:6px;padding:6px 9px;border-radius:8px;cursor:pointer}
.li.on{background:var(--card2)}.li .t{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.li .pn{color:var(--ac);width:13px}
.li .mo{opacity:0;color:var(--mu)}.li:hover .mo,.li.on .mo{opacity:1}
.ib{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:7px;color:var(--mu)}
.ib:hover,.ib.on{background:var(--hov);color:var(--tx)}
.empty{color:var(--mu);padding:8px 10px;font-size:13px}
.foot{display:flex;align-items:center;gap:6px;border-top:1px solid var(--bd);padding-top:9px;margin-top:6px}
.conn{display:flex;align-items:center;gap:8px;flex:1;min-width:0;padding:5px 8px;border-radius:8px;color:var(--mu)}
.conn:hover{background:var(--hov)}.conn i{width:8px;height:8px;border-radius:50%;background:var(--er);flex:none}.conn.ok i{background:var(--ok)}
.conn span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}

/* main */
.main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
.top{display:flex;align-items:center;gap:8px;height:50px;padding:0 14px;flex:none}
.top .ttl{min-width:0;display:flex;flex-direction:column;line-height:1.25}
.top .ttl b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ws{color:var(--mu);font-size:11.5px;cursor:pointer;font-family:var(--mono)}.ws:hover{color:var(--ac)}
.crumb{color:var(--mu);cursor:pointer}.crumb:hover{color:var(--tx)}
.sidebtn{display:none}
.msgs{flex:1;overflow:auto;padding:8px 18px 20px}
.col{max-width:760px;margin:0 auto;width:100%}
.dock{padding:0 18px 12px}
.fine{text-align:center;color:var(--mu);font-size:11.5px;padding-top:7px}
.m{margin:18px 0}
.m.u{display:flex;flex-direction:column;align-items:flex-start;gap:6px}
.ub{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:9px 14px;white-space:pre-wrap;word-break:break-word;max-width:100%}
.fchips{display:flex;gap:6px;flex-wrap:wrap}
.fc{display:inline-flex;gap:6px;align-items:center;font-size:12px;background:var(--card2);border:1px solid var(--bd);border-radius:8px;padding:3px 8px}
.ab{font-family:var(--ft);font-size:15.5px;line-height:1.68;word-break:break-word}
.ab p{margin:.55em 0}.ab h1,.ab h2,.ab h3,.ab h4{font-family:var(--ft);margin:1.1em 0 .4em;line-height:1.3}
.ab h1{font-size:1.5em}.ab h2{font-size:1.28em}.ab h3{font-size:1.12em}
.ab ul,.ab ol{margin:.5em 0;padding-left:1.3em}.ab li{margin:.2em 0}
.ab blockquote{margin:.7em 0;padding:.1em 1em;border-left:3px solid var(--ac);color:var(--mu)}
.ab hr{border:0;border-top:1px solid var(--bd);margin:1.2em 0}
.ab code{font:12.8px var(--mono);background:var(--card2);padding:1.5px 5px;border-radius:5px}
.tw{overflow:auto;margin:.7em 0}.ab table{border-collapse:collapse;font-size:14px;font-family:var(--sans)}
.ab th,.ab td{border:1px solid var(--bd);padding:5px 10px}.ab th{background:var(--card)}
.cb{margin:.8em 0;border:1px solid var(--bd);border-radius:10px;overflow:hidden;background:var(--code);font-family:var(--sans)}
.cbh{display:flex;align-items:center;justify-content:space-between;padding:4px 6px 4px 12px;font-size:12px;color:var(--mu);border-bottom:1px solid var(--bd)}
.cbh button{padding:3px 8px;border-radius:6px}.cbh button:hover{background:var(--hov);color:var(--tx)}
.cb pre{margin:0;padding:12px 14px;overflow:auto}.cb pre code{background:none;padding:0;font:12.8px/1.6 var(--mono);color:var(--tx)}
.tk-c{color:#7d8b7a;font-style:italic}.tk-s{color:#a5c98a}.tk-n{color:#e5a06b}.tk-k{color:#d97b9c}.tk-f{color:#79b8e8}
[data-theme=light] .tk-s{color:#4c8a2e}[data-theme=light] .tk-k{color:#b9366a}[data-theme=light] .tk-f{color:#1f6fb0}[data-theme=light] .tk-n{color:#b8651d}
.think{margin:.6em 0;border:1px dashed var(--bd);border-radius:10px;padding:6px 12px;font:13px/1.55 var(--sans);color:var(--mu)}
.think summary{cursor:pointer}.think div{margin-top:6px;white-space:pre-wrap}
.acts{display:flex;gap:2px;margin-top:4px;opacity:0;transition:opacity .15s}.m:hover .acts,.m.last .acts{opacity:1}
.cur{display:inline-block;width:8px;height:15px;background:var(--ac);vertical-align:-2px;margin-left:2px;animation:bl 1s steps(2) infinite}
@keyframes bl{50%{opacity:0}}
.dots{display:inline-flex;gap:4px;padding:6px 0}.dots i{width:7px;height:7px;border-radius:50%;background:var(--ac);animation:dd 1.2s infinite}.dots i:nth-child(2){animation-delay:.15s}.dots i:nth-child(3){animation-delay:.3s}
@keyframes dd{0%,60%,100%{opacity:.25}30%{opacity:1}}
.note{text-align:center;color:var(--mu);font-size:12.5px;margin:12px 0}

/* composer */
.comp{background:var(--card);border:1px solid var(--bd);border-radius:20px;padding:10px 12px 8px;box-shadow:0 2px 14px #0000001f}
.comp:focus-within{border-color:#8885}
.comp textarea{border:0;background:none;resize:none;padding:4px 4px 6px;max-height:240px;font-size:15px}
.crow{display:flex;align-items:center;gap:4px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:9px;color:var(--mu);font-size:13px}
.pill:hover{background:var(--hov);color:var(--tx)}.pill svg{width:14px;height:14px}
.send{width:32px;height:32px;border-radius:9px;background:var(--ac);color:#fff;display:inline-flex;align-items:center;justify-content:center;margin-left:4px}
.send:hover{filter:brightness(1.1)}.send.stop{background:var(--card2);color:var(--tx)}
#files{display:flex;gap:6px;flex-wrap:wrap}#files:not(:empty){padding-bottom:6px}

/* home */
.home{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px 18px 70px;background:var(--glow);overflow:auto}
.home>*{width:100%;max-width:700px}
.hello{display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:6px}
.hello h1{font:400 clamp(26px,4.4vw,40px)/1.15 var(--serif);margin:0;text-align:center}
.sub{text-align:center;color:var(--mu);margin-bottom:26px;font-size:13px}
.chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:16px}
.chip{display:inline-flex;gap:7px;align-items:center;padding:7px 13px;border:1px solid var(--bd);border-radius:999px;color:var(--mu);font-size:13px}
.chip:hover{color:var(--tx);border-color:var(--mu)}
.wsrow{display:flex;gap:8px;align-items:center;margin-bottom:12px;color:var(--mu)}.wsrow input{font-family:var(--mono);font-size:12.5px}

/* buttons, menus, modals */
.btn{display:inline-flex;align-items:center;gap:7px;padding:7px 14px;border-radius:9px;border:1px solid var(--bd);background:var(--card)}
.btn:hover{background:var(--card2)}.btn.pr{background:var(--ac);border-color:var(--ac);color:#fff}.btn.dg{color:var(--er)}
#menu{position:fixed;z-index:50;min-width:230px;max-width:340px;max-height:70vh;overflow:auto;background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:5px;box-shadow:0 10px 36px #0006;display:none}
#menu.on{display:block}#menu hr{border:0;border-top:1px solid var(--bd);margin:4px 0}
.mi{display:flex;align-items:center;gap:9px;width:100%;padding:7px 10px;border-radius:7px}.mi:hover{background:var(--hov)}
.mi span{flex:1}.mi small{color:var(--mu);font-size:11.5px;max-width:150px;text-align:right}.mi.dg{color:var(--er)}.mi .ck{color:var(--ac)}
#overlay{position:fixed;inset:0;background:#0008;z-index:40;display:none;align-items:flex-start;justify-content:center;padding-top:12vh}#overlay.on{display:flex}
.modal{background:var(--bg);border:1px solid var(--bd);border-radius:16px;padding:18px;width:min(460px,92vw);max-height:78vh;overflow:auto;box-shadow:0 20px 60px #0008}
.modal.wide{width:min(720px,94vw)}.modal h3{margin:0 0 12px;font:600 18px var(--serif)}.mb{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
.modal label{display:block;margin:10px 0 4px;color:var(--mu);font-size:12.5px}
.sbox{display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--bd);padding-bottom:10px}.sbox input{border:0;background:none;font-size:16px}
.sres{margin-top:8px}.sr{display:flex;gap:10px;width:100%;padding:8px;border-radius:8px;align-items:flex-start}.sr:hover{background:var(--hov)}
.sr span{display:flex;flex-direction:column;min-width:0}.sr small{color:var(--mu);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.viewer pre{margin:0;max-height:62vh;overflow:auto;background:var(--code);border:1px solid var(--bd);border-radius:10px;padding:12px;font:12.5px/1.55 var(--mono)}
#toast{position:fixed;bottom:22px;left:50%;transform:translate(-50%,20px);background:var(--card2);border:1px solid var(--bd);padding:8px 16px;border-radius:10px;opacity:0;pointer-events:none;transition:.2s;z-index:60}
#toast.on{opacity:1;transform:translate(-50%,0)}

/* pages: projects + customize */
.page{flex:1;overflow:auto;padding:10px 26px 40px}.pg{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;max-width:1000px;margin:0 auto}
.pcard,.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px 16px}.pcard{cursor:pointer}.pcard:hover{border-color:var(--mu)}
.pcard p{color:var(--mu);margin:.3em 0 .8em;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.pcard small{color:var(--mu)}
.two{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:24px;max-width:1040px;margin:0 auto;width:100%}
.two h2{font:400 28px var(--serif);margin:6px 0}.card h4{margin:0 0 6px;display:flex;align-items:center;justify-content:space-between}.card p{margin:0;color:var(--mu);white-space:pre-wrap}
.pchats{margin-top:20px}.row{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px}
.cust{flex:1;display:flex;min-height:0}.cn{width:200px;flex:none;padding:6px 10px;display:flex;flex-direction:column;gap:2px}
.cn button{padding:8px 12px;border-radius:8px;color:var(--mu)}.cn button.on,.cn button:hover{background:var(--hov);color:var(--tx)}
.cb2{flex:1;overflow:auto;padding:0 30px 50px}.cb2>*{max-width:680px}.cb2 h2{font:400 28px var(--serif);margin:6px 0 4px}
.f{margin:14px 0}.f>label{display:block;font-weight:600;margin-bottom:5px}.f small{display:block;color:var(--mu);margin-top:4px}
.rng{display:flex;gap:10px;align-items:center}.rng output{width:56px;text-align:right;font-family:var(--mono);color:var(--mu)}
.pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin:12px 0}
.pc{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:9px 12px;display:flex;flex-direction:column}.pc small{color:var(--mu)}.pc.on{border-color:var(--ac)}
.sw{display:inline-flex;gap:8px}.sw button{width:26px;height:26px;border-radius:50%;border:2px solid transparent}.sw button.on{border-color:var(--tx)}
.ok{color:var(--ok)}.er{color:var(--er)}.mrow{display:grid;grid-template-columns:120px 1fr 28px;gap:8px;margin:8px 0;align-items:start}

/* code agent */
.cwrap{flex:1;display:flex;min-height:0}.cmain{flex:1;display:flex;flex-direction:column;min-width:0}
.cpanel{width:290px;flex:none;border-left:1px solid var(--bd);overflow:auto;padding:10px 12px;background:var(--side)}
.ph{display:flex;align-items:center;justify-content:space-between;font-weight:600;margin:8px 0 6px}
.todo{list-style:none;margin:0;padding:0}.todo li{display:flex;gap:8px;align-items:flex-start;padding:2px 0;font-size:13px}
.todo li i{width:14px;height:14px;border:1.5px solid var(--mu);border-radius:50%;margin-top:3px;flex:none}.todo li.d{color:var(--mu);text-decoration:line-through}.todo li.d svg{color:var(--ok);margin-top:3px}
.tree .tr1{display:flex;align-items:center;gap:6px;padding:2px 4px;border-radius:5px;cursor:pointer;font:12.5px var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tree .tr1:hover{background:var(--hov)}.tree .tr1.ch{color:var(--ac)}.tree svg{width:13px;height:13px}
.tc{border:1px solid var(--bd);border-radius:10px;background:var(--card);margin:10px 0;overflow:hidden;font:13px var(--sans)}
.tch{display:flex;gap:8px;align-items:center;padding:7px 11px}.tch b{font-weight:600}.tch code{font:12px var(--mono);color:var(--mu);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:none;padding:0}
.tch svg{color:var(--ac)}.tc pre,.tc .fin,.tc ul{margin:0;padding:8px 12px;border-top:1px solid var(--bd);font:12px/1.5 var(--mono);overflow:auto;max-height:260px;background:var(--code);white-space:pre}
.tc .fin{font:13px var(--sans);white-space:normal;background:none}.tc .todo{font-family:var(--sans)}
.tc details summary{padding:5px 12px;cursor:pointer;color:var(--mu);font-size:12px;border-top:1px solid var(--bd)}
.dl{display:block}.dl.a{background:#2ea04322}.dl.r{background:#e5534b22}
.spin{width:12px;height:12px;border:2px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:sp .8s linear infinite;margin-left:auto}@keyframes sp{to{transform:rotate(1turn)}}
.tr{border:1px solid var(--bd);border-radius:8px;margin:-4px 0 6px 18px;font-size:12.5px}.tr summary{cursor:pointer;display:flex;gap:7px;align-items:center;padding:4px 10px;color:var(--mu);list-style:none}
.tr summary span{margin-left:auto}.tr.ok summary svg{color:var(--ok)}.tr.er summary svg{color:var(--er)}
.tr pre{margin:0;padding:8px 12px;border-top:1px solid var(--bd);font:12px/1.5 var(--mono);overflow:auto;max-height:280px;background:var(--code);white-space:pre-wrap;word-break:break-word}
.appr{border:1.5px solid var(--ac);border-radius:12px;padding:4px 12px 12px;margin:14px 0;background:var(--card)}
.appr .tc{border:0;margin:6px 0}.appr .bs{display:flex;gap:8px;flex-wrap:wrap}

@media(max-width:820px){.side{position:fixed;z-index:30;left:-290px;top:0;bottom:0;transition:left .2s}body.sb .side{left:0}.sidebtn{display:inline-flex}.cpanel{display:none}.two{grid-template-columns:1fr}.cn{width:130px}.cb2{padding:0 14px 40px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ===================== effects, backgrounds, effort colours ===================== */
#bgl{position:fixed;inset:0;z-index:0;display:none;background-size:cover;background-position:center}
body.wp #bgl{display:block}
#bgl::after{content:"";position:absolute;inset:0;background:var(--dimc,#000);opacity:var(--dim,0)}
#bgl.aur{background-size:160% 160%;animation:drift 38s ease-in-out infinite alternate}
@keyframes drift{from{background-position:0% 0%}to{background-position:100% 100%}}
#app{position:relative;z-index:1}
body.wp:not(.gl) .side,body.wp:not(.gl) .main{background:color-mix(in srgb,var(--bg) var(--pn,75%),transparent)}
body.wp .side{backdrop-filter:blur(18px) saturate(1.4);-webkit-backdrop-filter:blur(18px) saturate(1.4)}
body.wp:not(.gl) :is(.comp,.ub,.card,.pcard,.pc,.tc,.cpanel,.seg,.think,.tr){background:color-mix(in srgb,var(--card) var(--pn,75%),transparent);backdrop-filter:blur(14px)}
body.wp:not(.gl) .cb{background:color-mix(in srgb,var(--code) var(--pn,75%),transparent)}

.mi svg{color:var(--c,inherit)}
.comp{position:relative;transition:border-color .2s,box-shadow .25s}
.comp:focus-within{border-color:color-mix(in srgb,var(--ef,var(--ac)) 60%,var(--bd));box-shadow:0 0 0 3px color-mix(in srgb,var(--ef,var(--ac)) 20%,transparent),0 2px 14px #0000001f}
.pill.ef{color:var(--ef)}
.send{background:var(--ef,var(--ac));color:var(--efc,#fff);transition:transform .15s,filter .15s,background .25s}
.send:hover{transform:scale(1.07)}.send:active{transform:scale(.92)}.send.stop{background:var(--card2);color:var(--tx)}
.lv{display:inline-flex;gap:2px;align-items:flex-end;margin:0 2px}
.lv i{width:3px;height:5px;background:var(--bd);border-radius:1px;transition:background .2s}
.lv i:nth-child(2){height:7px}.lv i:nth-child(3){height:9px}.lv i:nth-child(4){height:11px}.lv i:nth-child(5){height:13px}
.lv i.on{background:var(--ef)}
.comp[data-ef=high]{animation:glow 3.4s ease-in-out infinite}
.comp[data-ef=extra]{animation:glow 2.2s ease-in-out infinite}
@keyframes glow{50%{box-shadow:0 0 0 1px color-mix(in srgb,var(--ef) 40%,transparent),0 0 18px color-mix(in srgb,var(--ef) 30%,transparent)}}
.comp[data-ef=ultracode]::after{content:"";position:absolute;inset:-1px;border-radius:inherit;padding:2px;pointer-events:none;
 background:linear-gradient(100deg,#e056c8,#f2994a,#f2c94c,#6fcf97,#6ea8fe,#e056c8);background-size:300% 100%;animation:efb 5s linear infinite;
 -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}
@keyframes efb{to{background-position:300% 0}}
.efleg{display:flex;gap:8px;flex-wrap:wrap}.efleg span{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;border:1px solid var(--bd);font-size:12.5px}.efleg i{width:9px;height:9px;border-radius:50%}

/* motion */
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes fade{from{opacity:0}to{opacity:1}}
@keyframes pop{from{opacity:0;transform:scale(.94) translateY(4px)}to{opacity:1;transform:none}}
@keyframes sparkin{from{transform:rotate(-120deg) scale(.4);opacity:0}to{transform:none;opacity:1}}
@keyframes spin{to{transform:rotate(1turn)}}
.m.new{animation:rise .38s cubic-bezier(.2,.8,.2,1)}
.main.enter{animation:fade .28s ease}
.hello h1{animation:rise .7s cubic-bezier(.2,.8,.2,1) both}.hello svg{animation:sparkin .9s cubic-bezier(.2,.9,.3,1.2) both}
.sub,.chips,.home .comp{animation:rise .7s .12s cubic-bezier(.2,.8,.2,1) both}
.chip{transition:transform .15s,border-color .15s,color .15s}.chip:hover{transform:translateY(-2px)}
.nav,.li,.btn,.ib,.pill,.pc,.pcard{transition:background .15s,transform .12s,color .15s,border-color .15s}
.btn:active,.pill:active,.nav:active{transform:scale(.97)}.pcard:hover,.pc:hover{transform:translateY(-1px)}
#menu.on{animation:pop .14s ease-out;transform-origin:bottom left}.modal{animation:pop .2s cubic-bezier(.2,.8,.2,1)}#overlay.on{animation:fade .15s}
.brand.busy svg{animation:spin 2.6s linear infinite}
.tc{animation:pop .25s ease-out}.appr{animation:pop .25s ease-out}
.anim-off *,.anim-off *::before,.anim-off *::after{animation:none!important;transition:none!important}
html.anim-sub *:not(.dots i):not(.cur):not(.spin):not(.pinch *),html.anim-sub *::after{animation-iteration-count:1!important}

/* skin picker + appearance page */
.skg{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:10px;margin:8px 0 4px}
.sk{border:2px solid var(--bd);border-radius:12px;padding:6px;background:var(--card);transition:transform .15s,border-color .15s}.sk:hover{transform:translateY(-2px)}
.sk.on{border-color:var(--ac)}.sk .sw3{display:flex;height:42px;border-radius:7px;overflow:hidden;margin-bottom:6px;box-shadow:inset 0 0 0 1px #0002}.sk .sw3 i{flex:1}.sk .sw3 i:first-child{flex:1.4}.sk b{display:block;font-size:12.5px;padding:0 2px}
.sg{font-size:12px;color:var(--mu);margin:14px 0 2px;text-transform:uppercase;letter-spacing:.06em}
.wallg{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:8px;margin:8px 0}
.wl{height:56px;border-radius:10px;border:2px solid var(--bd);display:flex;align-items:flex-end;padding:4px 7px;font-size:11.5px;color:#fff;text-shadow:0 1px 3px #000a;transition:transform .15s}.wl:hover{transform:translateY(-2px)}.wl.on{border-color:var(--ac)}
.chk{display:flex;align-items:center;gap:10px;margin:10px 0;cursor:pointer}.chk input{width:auto;accent-color:var(--ac)}
.dstat{display:flex;align-items:center;gap:8px;margin:12px 0;padding:8px 12px;border:1px solid var(--bd);border-radius:10px;background:var(--card)}.dstat i{width:9px;height:9px;border-radius:50%;background:var(--mu)}.dstat.connected i{background:var(--ok)}.dstat.error i{background:var(--er)}.dstat.waiting i{background:#e5b94a}

/* ===================== Pinch, the pixel crab ===================== */
body.crab .dock{padding-top:34px}body.crab .home .comp{margin-top:38px}
.pinch{position:absolute;left:calc(var(--px,.93)*(100% - var(--cs,64px)));bottom:100%;width:var(--cs,64px);margin-bottom:-3px;cursor:pointer;z-index:3;user-select:none;transition:left var(--wd,.8s) linear}
.pinch svg{width:100%;display:block;overflow:visible}.pinch svg g{transform-box:fill-box;transform-origin:center}
.pb{position:absolute;right:calc(100% + 2px);bottom:42%;background:var(--card2);color:var(--tx);border:2px solid var(--tx);border-radius:3px;font:600 11px/1.25 var(--mono);padding:3px 7px;white-space:nowrap;box-shadow:2px 2px 0 #0006;animation:pop .18s steps(3)}
.pb::after{content:"";position:absolute;right:-6px;bottom:5px;width:6px;height:6px;background:var(--card2);border-top:2px solid var(--tx);border-right:2px solid var(--tx);transform:rotate(45deg) translate(-2px,2px)}
.cr-eyes{animation:blink 4.6s steps(1) infinite}
@keyframes blink{0%,93%,100%{transform:scaleY(1)}94%,97%{transform:scaleY(.12)}}
.m-idle .cr-all{animation:bob 2.6s steps(1) infinite}
@keyframes bob{0%,50%{transform:none}50.01%,100%{transform:translateY(-1px)}}
.m-idle .cr-cl-r{animation:wave 8s steps(1) infinite}
@keyframes wave{0%,86%{transform:none}88%,92%{transform:translateY(-2px)}90%,94%{transform:translateY(-1px)}95%,100%{transform:none}}
.m-think .cr-eyes{animation:none;transform:translateY(-1px)}.m-think .cr-cl-r{animation:tap .7s steps(1) infinite}
@keyframes tap{0%,50%{transform:translate(-3px,-1px)}50.01%,100%{transform:translate(-3px,0)}}
.m-write .cr-cl-l{animation:type .28s steps(1) infinite}.m-write .cr-cl-r{animation:type .28s steps(1) infinite reverse}
@keyframes type{0%,50%{transform:translateY(0)}50.01%,100%{transform:translateY(2px)}}
.m-write .cr-eyes{animation:none;transform:translateY(1px)}
.m-run .cr-all{animation:scut .5s steps(1) infinite}@keyframes scut{0%,50%{transform:translateX(-2px)}50.01%,100%{transform:translateX(2px)}}
.m-run .cr-lgA{animation:leg .25s steps(1) infinite}.m-run .cr-lgB{animation:leg .25s steps(1) infinite reverse}
@keyframes leg{0%,50%{transform:translateY(0)}50.01%,100%{transform:translateY(-1px)}}
.m-run .cr-cl{transform:translateY(-1px)}
.m-look .cr-eyes{animation:look 1.5s steps(1) infinite}@keyframes look{0%,50%{transform:translateX(-1px)}50.01%,100%{transform:translateX(1px)}}
.m-ask .cr-cl{animation:raise .5s steps(1) infinite}@keyframes raise{0%,50%{transform:translateY(-2px)}50.01%,100%{transform:translateY(0)}}
.m-happy .cr-all{animation:hop .6s steps(1) 5}@keyframes hop{0%{transform:none}30%{transform:translateY(-4px)}60%{transform:translateY(-4px)}80%{transform:translateY(-1px)}100%{transform:none}}
.m-happy .cr-cl{transform:translateY(-2px)}
.m-sad .cr-all{animation:shake .2s steps(1) 8}@keyframes shake{0%,50%{transform:translateX(-1px)}50.01%,100%{transform:translateX(1px)}}.m-sad .cr-cl{transform:translateY(1px)}
.m-sleep .cr-all{animation:bob 4s steps(1) infinite}.m-sleep .cr-eyes{animation:none}
.cf{animation:cf 1.1s steps(6) forwards}@keyframes cf{from{transform:translate(0,0);opacity:1}to{transform:translate(var(--dx),var(--dy));opacity:0}}
.zz{animation:zz 2.4s steps(6) infinite;font:700 3px var(--mono);fill:var(--tx)}@keyframes zz{from{transform:translate(0,0);opacity:1}to{transform:translate(3px,-5px);opacity:0}}
.drop{animation:drop 1s steps(5) infinite}@keyframes drop{from{transform:translateY(0);opacity:1}to{transform:translateY(4px);opacity:0}}

/* ===================== themes ===================== */
html[data-skin=midnight]{--bg:#12141f;--side:#0d0f18;--card:#1b1e2e;--card2:#252a3f;--bd:#2d3350;--tx:#e6e9f5;--mu:#8b93b5;--ac:#7aa2f7;--code:#0e1019}
html[data-skin=nord]{--bg:#2e3440;--side:#272c36;--card:#3b4252;--card2:#434c5e;--bd:#4c566a;--tx:#eceff4;--mu:#a3adc2;--ac:#88c0d0;--code:#252a33}
html[data-skin=dracula]{--bg:#282a36;--side:#21222c;--card:#343746;--card2:#3e4254;--bd:#4a4e63;--tx:#f8f8f2;--mu:#9aa0bd;--ac:#bd93f9;--code:#1e1f29}
html[data-skin=solarized]{--bg:#002b36;--side:#00212b;--card:#073642;--card2:#0d4452;--bd:#1a5666;--tx:#eee8d5;--mu:#93a1a1;--ac:#b58900;--code:#001f27}
html[data-skin=rose]{--bg:#191724;--side:#13111c;--card:#1f1d2e;--card2:#26233a;--bd:#393552;--tx:#e0def4;--mu:#908caa;--ac:#eb6f92;--code:#12101a}
html[data-skin=forest]{--bg:#1a211b;--side:#141a15;--card:#222c24;--card2:#2b372d;--bd:#34443a;--tx:#e8efe6;--mu:#96a897;--ac:#8fbf7f;--code:#121712}

/* Terminal (CRT) */
html[data-skin=terminal]{--bg:#050805;--side:#030503;--card:#0a120a;--card2:#102010;--bd:#1c4a22;--tx:#4dff7a;--mu:#2fa84d;--ac:#4dff7a;--ok:#4dff7a;--er:#ff5f5f;--code:#020402;--hov:#4dff7a14;--sans:var(--mono);--ft:var(--mono)}
html[data-skin=terminal] body{text-shadow:0 0 5px #4dff7a55}
html[data-skin=terminal] body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:90;background:repeating-linear-gradient(#0000 0 2px,#0000003a 2px 3px);animation:crt 9s linear infinite}
@keyframes crt{50%{opacity:.82}}
html[data-skin=terminal] .send{color:#001a08}

/* Liquid Glass */
html[data-skin=glass]{--bg:#0b1020;--side:#ffffff14;--card:#ffffff1f;--card2:#ffffff30;--bd:#ffffff38;--tx:#fff;--mu:#ffffffb8;--ac:#8ccfff;--code:#0a0f1e80;--hov:#ffffff1c;
 --sans:-apple-system,"SF Pro Display","Segoe UI Variable","Segoe UI",Inter,sans-serif;--ft:var(--sans)}
html[data-skin=glass] :is(.side,.comp,.ub,.card,.pcard,.pc,.tc,.cpanel,.seg,.think,.tr,.cb,.sk,.dstat,.modal,#menu,.appr,.chip,.btn,.pill){backdrop-filter:blur(26px) saturate(1.9);-webkit-backdrop-filter:blur(26px) saturate(1.9);box-shadow:inset 0 1px 0 #ffffff70,inset 0 -1px 0 #ffffff14,0 10px 34px #00000040}
html[data-skin=glass] :is(.comp,.modal,.card,.tc,.cb){border-radius:22px}html[data-skin=glass] :is(.ub){border-radius:20px}
html[data-skin=glass] .modal,html[data-skin=glass] #menu{background:#1b2540c0}
html[data-skin=glass] .side{border-right-color:#ffffff26}
html[data-skin=glass] .send{box-shadow:inset 0 1px 0 #fff9,0 4px 16px color-mix(in srgb,var(--ef,var(--ac)) 60%,transparent)}
html[data-skin=glass] .ab,html[data-skin=glass] .ub,html[data-skin=glass] h1{text-shadow:0 1px 8px #0006}

/* macOS */
html[data-skin=macos]{--bg:#f6f6f8;--side:#e9e9ee;--card:#fff;--card2:#efeff3;--bd:#d6d6dc;--tx:#1d1d1f;--mu:#6e6e73;--ac:#0a84ff;--code:#f3f3f7;--hov:#0000000c;
 --sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;--ft:var(--sans)}
html[data-skin=macos] .side{background:#e9e9eeee;backdrop-filter:blur(30px) saturate(1.8)}
html[data-skin=macos] .brand{font-family:var(--sans);font-weight:600;font-size:15px}
html[data-skin=macos] .brand::before{content:"";flex:none;width:12px;height:12px;border-radius:50%;background:#ff5f57;box-shadow:20px 0 0 #febc2e,40px 0 0 #28c840,inset 0 0 0 1px #0002;margin:0 54px 0 3px}
html[data-skin=macos] .li.on,html[data-skin=macos] .nav.on{background:var(--ac);color:#fff}html[data-skin=macos] .li.on .mo{color:#fff}
html[data-skin=macos] .comp{box-shadow:0 1px 3px #0002,0 8px 28px #0000001a;border-color:#d0d0d6}
html[data-skin=macos] .btn.pr,html[data-skin=macos] .send{box-shadow:inset 0 1px 0 #ffffff55,0 1px 2px #0003}

/* Windows 11 */
html[data-skin=win11]{--bg:#f3f3f3;--side:#eaeaea;--card:#fff;--card2:#f7f7f7;--bd:#e1e1e1;--tx:#1a1a1a;--mu:#616161;--ac:#0067c0;--code:#f7f7f7;--hov:#0000000a;
 --sans:"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;--ft:var(--sans)}
html[data-skin=win11] .li.on::before{content:"";width:3px;height:16px;background:var(--ac);border-radius:2px;margin-right:-2px}
html[data-skin=win11] .top::after{content:"—      ☐      ✕";font-size:12px;color:var(--mu);white-space:pre;margin-right:2px}
html[data-skin=win11] .comp,html[data-skin=win11] .card{box-shadow:0 1px 2px #0000000f}

/* shared "window" chrome for the retro Windows skins */
html[data-skin=win98] #app,html[data-skin=winxp] #app,html[data-skin=vista] #app{margin:8px;height:calc(100vh - 16px);padding-top:30px}
html[data-skin=win98] #app::before,html[data-skin=winxp] #app::before,html[data-skin=vista] #app::before{content:"KoboldCode";position:absolute;left:0;right:0;top:0;height:30px;padding:0 10px;font:700 13px/30px var(--sans);z-index:2;pointer-events:none}
html[data-skin=win98] #app::after,html[data-skin=winxp] #app::after,html[data-skin=vista] #app::after{position:absolute;top:5px;right:6px;z-index:3;pointer-events:none;font:700 12px/18px var(--sans);white-space:pre}

/* Windows 98 */
html[data-skin=win98]{--bg:#c0c0c0;--side:#c0c0c0;--card:#c0c0c0;--card2:#dfdfdf;--bd:#808080;--tx:#000;--mu:#404040;--ac:#000080;--ok:#006400;--er:#a00000;--code:#fff;--hov:#00000014;
 --sans:Tahoma,"MS Sans Serif",Verdana,Arial,sans-serif;--ft:var(--sans);--mono:"Courier New",Courier,monospace;
 --raise:inset -1px -1px #0a0a0a,inset 1px 1px #fff,inset -2px -2px #808080,inset 2px 2px #dfdfdf;--sunk:inset -1px -1px #fff,inset 1px 1px #808080,inset -2px -2px #dfdfdf,inset 2px 2px #0a0a0a}
html[data-skin=win98] body{background:#008080;font-size:13px}
html[data-skin=win98] #app{background:#c0c0c0;box-shadow:var(--raise)}
html[data-skin=win98] #app::before{left:3px;right:3px;top:3px;height:20px;line-height:20px;padding:0 6px;background:linear-gradient(90deg,#000080,#1084d0);color:#fff}
html[data-skin=win98] #app::after{content:"_  ▫  ×";top:5px;right:6px;background:#c0c0c0;color:#000;box-shadow:var(--raise);line-height:14px;padding:0 6px}
html[data-skin=win98] :is(.btn,.pill,.send,.ib,.chip,.pc,.sk,.seg button,.nav){border:0;box-shadow:var(--raise);background:#c0c0c0;color:#000}
html[data-skin=win98] :is(.btn:active,.pill:active,.send:active){box-shadow:var(--sunk)}
html[data-skin=win98] :is(input,textarea,select,.comp,.list,.msgs,.home,.tree,.cb,.cpanel){background:#fff;color:#000;border:0;box-shadow:var(--sunk)}
html[data-skin=win98] .comp textarea{box-shadow:none}
html[data-skin=win98] :is(.msgs,.list){margin:2px 6px}html[data-skin=win98] .main{padding-bottom:4px}
html[data-skin=win98] .btn.pr,html[data-skin=win98] .seg button.on,html[data-skin=win98] .li.on,html[data-skin=win98] .nav.on{background:#000080;color:#fff;box-shadow:var(--raise)}
html[data-skin=win98] .li.on{box-shadow:none}html[data-skin=win98] .li.on .mo{color:#fff}
html[data-skin=win98] .ub{background:#ffffe1;border:1px solid #000;color:#000}
html[data-skin=win98] :is(.card,.tc,.modal,#menu,.appr){border:0;box-shadow:var(--raise);background:#c0c0c0}
html[data-skin=win98] .tc pre,html[data-skin=win98] .tr pre{background:#fff;color:#000}
html[data-skin=win98] .send{color:#000}html[data-skin=win98] .lv i{background:#808080}
html[data-skin=win98] .side,html[data-skin=win98] .top{border:0}html[data-skin=win98] .hello h1{font-weight:700}

/* Windows XP */
html[data-skin=winxp]{--bg:#ece9d8;--side:#d6dff7;--card:#fff;--card2:#f4f3ee;--bd:#7f9db9;--tx:#000;--mu:#4a4a4a;--ac:#316ac5;--code:#fff;--hov:#316ac51c;
 --sans:Tahoma,Verdana,"Segoe UI",sans-serif;--ft:var(--sans)}
html[data-skin=winxp] body{background:radial-gradient(ellipse 130% 55% at 25% 112%,#4aa02c 0 62%,#0000 63%),linear-gradient(#2a62d0,#8cbcf4 72%)}
html[data-skin=winxp] #app{border:3px solid #0831d9;border-top:0;border-radius:9px 9px 3px 3px;background:var(--bg);box-shadow:0 8px 30px #0007}
html[data-skin=winxp] #app::before{background:linear-gradient(#0a5cf0,#0054e3 10%,#3d95ff 40%,#0a59e6 86%,#0831d9);color:#fff;text-shadow:1px 1px #0f1089;border-radius:6px 6px 0 0;margin:0 -3px;padding-left:13px;font-family:"Trebuchet MS",Tahoma,sans-serif}
html[data-skin=winxp] #app::after{content:"–  ▢  ✕";color:#fff;top:6px;right:4px;letter-spacing:2px;text-shadow:1px 1px #0f1089;background:none}
html[data-skin=winxp] .side{background:linear-gradient(180deg,#7ba2e7,#6375d6);color:#fff;--tx:#fff;--mu:#dbe6ff;--hov:#ffffff30;--bd:#ffffff55;--card:#ffffff22;--card2:#ffffff40}
html[data-skin=winxp] .btn,html[data-skin=winxp] .pill{border-radius:3px;border:1px solid #003c74;background:linear-gradient(#fff,#ecebe5 85%,#d6d0c5)}
html[data-skin=winxp] .btn.pr,html[data-skin=winxp] .send{background:linear-gradient(#5cc045,#2f8f1e);border:1px solid #1d5a12;color:#fff;border-radius:4px}
html[data-skin=winxp] .comp{border-radius:4px}

/* Windows Vista (Aero) */
html[data-skin=vista]{--bg:#eaf1f8;--side:#ffffff4d;--card:#ffffffc4;--card2:#ffffffe0;--bd:#8aa4c4;--tx:#0e2233;--mu:#46627c;--ac:#2a7bd8;--code:#f6f9fd;--hov:#2a7bd81a;
 --sans:"Segoe UI",Frutiger,Tahoma,sans-serif;--ft:var(--sans)}
html[data-skin=vista] #app{border-radius:9px;background:linear-gradient(#a8c8ee80,#6d9bd080);backdrop-filter:blur(16px) saturate(1.5);border:1px solid #ffffffb0;box-shadow:0 0 0 1px #000a,0 14px 50px #000a;padding:30px 7px 7px}
html[data-skin=vista] #app::before{color:#0b1c2c;text-shadow:0 0 10px #fff,0 0 14px #fff,0 0 4px #fff}
html[data-skin=vista] #app::after{content:"✕";top:0;right:10px;width:46px;height:20px;line-height:19px;text-align:center;color:#fff;background:linear-gradient(#e8a095,#c9301a 50%,#a31500 52%,#d8492e);border:1px solid #521;border-top:0;border-radius:0 0 6px 6px;box-shadow:inset 0 1px 0 #fff8}
html[data-skin=vista] .side{background:#ffffff40;border-radius:3px 0 0 3px}html[data-skin=vista] .main{background:#ffffffe0;border-radius:0 3px 3px 0}
html[data-skin=vista] .btn,html[data-skin=vista] .pill{border:1px solid #707070;border-radius:3px;background:linear-gradient(#f2f2f2,#ebebeb 48%,#dddddd 52%,#cfcfcf)}
html[data-skin=vista] .btn.pr,html[data-skin=vista] .send{background:linear-gradient(#7fb4f0,#3c86e0 48%,#1f6ccd 52%,#3d95f2);border:1px solid #1d4f91;color:#fff;box-shadow:inset 0 1px 0 #fff9}

/* square, pixel-ish corners for the retro skins (status dots and the crab stay round) */
html[data-skin=win98] *:not(i):not(.spin):not(.pinch):not(.pinch *),html[data-skin=terminal] *:not(i):not(.spin):not(.pinch):not(.pinch *){border-radius:0!important}
@media(max-width:820px){html[data-skin=win98] #app,html[data-skin=winxp] #app,html[data-skin=vista] #app{margin:0;height:100vh}}
body.wp .home{background:none}

/* Pinch everywhere */
.pinch.mini{position:static;display:inline-block;width:var(--cs,30px);margin:0;cursor:default;vertical-align:middle;transition:none}
.cr-look{transform:translate(calc(var(--ex,0)*1px),calc(var(--ey,0)*1px))}
.pinch.pbr .pb{right:auto;left:calc(100% + 2px)}.pinch.pbr .pb::after{right:auto;left:-6px;transform:rotate(-135deg) translate(-2px,2px)}
.m-walk .cr-lgA{animation:leg .3s steps(1) infinite}.m-walk .cr-lgB{animation:leg .3s steps(1) infinite reverse}.m-walk .cr-all{animation:wb .6s steps(1) infinite}.m-walk .cr-cl{transform:translateY(-1px)}
@keyframes wb{0%,50%{transform:none}50.01%,100%{transform:translateY(-1px)}}
.tw{animation:tw 1.4s steps(1) infinite}.tw2{animation-delay:.7s}@keyframes tw{0%,60%{opacity:1}61%,100%{opacity:0}}
.rb{animation:hue 3s linear infinite}@keyframes hue{to{filter:hue-rotate(360deg)}}
.m.a.av{display:grid;grid-template-columns:38px minmax(0,1fr);column-gap:8px}
.m.a.av .avt{grid-column:1;grid-row:1;padding-top:4px}.m.a.av .ab,.m.a.av .acts{grid-column:2}
.pe{display:flex;align-items:center;gap:8px}#toast{display:flex;align-items:center;gap:8px}
.crow{flex-wrap:nowrap;min-width:0}.pill{min-width:0;flex:0 1 auto}.pt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}.send,.ib{flex:none}
body.avon .tr{margin-left:46px}
.pill.ef{flex-shrink:.15}

</style>
</head>
<body>
<div id="bgl"></div>
<div id="app"></div>
<div id="overlay" data-a="closebg"></div>
<div id="menu"></div>
<div id="toast"></div>
<input type="file" id="fi" multiple hidden>
<input type="file" id="pfi" multiple hidden>
<input type="file" id="imp" accept=".json" hidden>
<input type="file" id="bgf" accept="image/*" hidden>
<script>const TOKEN="__TOKEN__",DEFAULT_URL="__KOBOLD__";</script>
<script>

'use strict';
/* ============ utils ============ */
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uid=()=>Math.random().toString(36).slice(2,8)+Date.now().toString(36).slice(-4);
const est=s=>Math.ceil((s||'').length/3.4);
const pick=a=>a[Math.floor(Math.random()*a.length)];
const enc=encodeURIComponent;
const IC={plus:'<path d="M12 5v14M5 12h14"/>',search:'<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
folder:'<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
sliders:'<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
pin:'<path d="M12 17v5M8 3h8l-1.5 7L18 14H6l3.5-4z"/>',trash:'<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>',
edit:'<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
more:'<circle cx="5" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="19" cy="12" r="1.2"/>',
up:'<path d="M12 19V5M5 12l7-7 7 7"/>',stop:'<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
copy:'<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
file:'<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6"/>',
term:'<path d="m4 17 6-6-6-6M12 19h8"/>',check:'<path d="M20 6 9 17l-5-5"/>',x:'<path d="M18 6 6 18M6 6l12 12"/>',
dl:'<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',undo:'<path d="M9 14 4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3"/>',
chev:'<path d="m6 9 6 6 6-6"/>',menu:'<path d="M3 6h18M3 12h18M3 18h18"/>',
clip:'<path d="m21 12.5-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l9-9a3.7 3.7 0 0 1 5.2 5.2l-9 9a1.8 1.8 0 0 1-2.6-2.6l8.4-8.4"/>',
bolt:'<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',shield:'<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/>',
list:'<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',refresh:'<path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5"/>',
code:'<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',panel:'<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
gear:'<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/>',
chat:'<path d="M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-5.6A8 8 0 1 1 21 12z"/>',play:'<path d="M6 4l14 8-14 8z"/>'};
const ic=(n,c='')=>`<svg class="i ${c}" viewBox="0 0 24 24">${IC[n]||''}</svg>`;
const LOGO=(s=22)=>`<svg width="${s}" height="${s}" viewBox="0 0 24 24" style="flex:none">${[0,1,2,3,4,5,6,7].map(i=>`<path d="M12 12V${i%2?4.6:2.2}" stroke="var(--ac)" stroke-width="2.3" stroke-linecap="round" transform="rotate(${i*45} 12 12)"/>`).join('')}</svg>`;
const api=async(p,b)=>{const r=await fetch(p,{method:b?'POST':'GET',headers:{'X-KC':TOKEN,'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});return r.json()};
const download=(name,text,type='text/plain')=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000)};
function toast(m){const t=$('#toast'),bad=/fail|could not|invalid|too large|not a valid|binary|unknown|nothing to|first|stop the|looks binary/i.test(m);
t.innerHTML=(typeof S==='object'&&S&&S.settings&&S.settings.crab?miniCrab(bad?'sad':'happy',26):'')+'<span></span>';t.lastChild.textContent=m;t.classList.add('on');clearTimeout(toast.t);toast.t=setTimeout(()=>t.classList.remove('on'),2400)}
const getPath=(o,p)=>p.split('.').reduce((a,k)=>a?.[k],o);
const setPath=(o,p,v)=>{const k=p.split('.');let t=o;for(let i=0;i<k.length-1;i++)t=t[k[i]];t[k[k.length-1]]=v};
const titleFrom=t=>(t||'').replace(/\s+/g,' ').trim().slice(0,48)||'New chat';

/* ============ time of day ============ */
const bucket=(d=new Date())=>{const h=d.getHours();return h<5||h>=22?'night':h<8?'dawn':h<12?'morning':h<17?'day':'evening'};
const GREET={
night:["It's a late-night jam session.","It's a late-night jam session.","Burning the midnight oil{n}?","The quiet hours are the good ones{n}.","Night owl mode: engaged."],
dawn:["Up before the sun{n}?","Early start{n}. Coffee first?","A quiet beginning to the day."],
morning:["Good morning{n}.","Morning{n}. What's on the agenda?","A fresh day, a fresh idea{n}."],
day:["Good afternoon{n}.","Afternoon momentum{n}.","Midday brainstorm{n}?"],
evening:["Good evening{n}.","Golden hour thinking{n}?","Winding down or ramping up{n}?"]};
const CG={night:"Late-night build session{n}. What are we shipping?",dawn:"Early start{n}. What are we building?",morning:"Morning{n}. What are we building today?",day:"Afternoon{n}. What should we ship?",evening:"Evening{n}. One more feature before dinner?"};
const GLOW={night:'radial-gradient(ellipse 70% 50% at 50% 0%,#4a5bb040,transparent)',dawn:'radial-gradient(ellipse 70% 50% at 50% 0%,#e8907a38,transparent)',morning:'radial-gradient(ellipse 70% 50% at 50% 0%,#f2c46d30,transparent)',day:'radial-gradient(ellipse 70% 50% at 50% 0%,#6aa8d62e,transparent)',evening:'radial-gradient(ellipse 70% 50% at 50% 0%,#d9775544,transparent)'};
const CHIPS={
night:[['Quiet-hours project idea','Give me a small, satisfying project I can finish tonight.'],['Explain something fascinating','Explain something fascinating about space or the deep sea, in plain language.'],['Rubber-duck my bug','I have a bug I can\'t crack. Let me describe it and help me debug step by step.']],
dawn:[['Plan my day','Help me plan my day. I will tell you my tasks and you order them by priority.'],['Warm-up writing prompt','Give me a short creative writing warm-up prompt.'],['Daily learning nugget','Teach me one useful thing in five minutes.']],
morning:[['Plan my day','Help me plan my day. I will tell you my tasks and you order them by priority.'],['Draft an email','Help me draft a clear, friendly email.'],['Summarize and focus','I will paste notes; summarize them and tell me what to focus on first.']],
day:[['Debug this code','Help me debug some code. I will paste it along with the error.'],['Brainstorm ideas','Brainstorm 10 creative ideas for a project I describe.'],['Explain a concept','Explain a concept to me step by step with an analogy.']],
evening:[['Review my day','Help me reflect on my day and decide what to carry over to tomorrow.'],['Learn something new','Create a 20-minute lesson plan for a topic I pick.'],['Tell me a story','Write a short, atmospheric story with a twist ending.']]};

/* ============ effort, presets, templates ============ */
const EFFORT=[
{col:'#6ea8fe',on:'#10131a',id:'low',name:'Low',desc:'Fast, short answers',max:768,dt:0,steps:8,hint:'Be brief and direct. Skip preamble.'},
{col:'#6fcf97',on:'#10131a',id:'medium',name:'Medium',desc:'Balanced speed and depth',max:2048,dt:0,steps:16,hint:'Give a clear, well-structured answer.'},
{col:'#f2c94c',on:'#10131a',id:'high',name:'High',desc:'Careful step-by-step reasoning',max:4096,dt:-.1,steps:26,hint:'Reason step by step and double-check your work before answering.'},
{col:'#f2994a',on:'#10131a',id:'extra',name:'Extra',desc:'Deep analysis, edge cases, self-checks',max:6144,dt:-.2,steps:40,hint:'Think deeply: consider alternatives and edge cases, verify your answer, then respond thoroughly.'},
{col:'#e056c8',on:'#ffffff',id:'ultracode',name:'Ultracode',desc:'Max effort: plan, build, test, review',max:8192,dt:-.3,steps:60,hint:'Maximum effort. Start with a short plan, then produce complete production-quality code with error handling, then review your own output for bugs and fix them. In agent mode, run and test the code before finishing.'}];
const TPL={
chatml:{n:'ChatML (Qwen, Hermes…)',s:'<|im_start|>system\n{}<|im_end|>\n',u:'<|im_start|>user\n{}<|im_end|>\n',a:'<|im_start|>assistant\n{}<|im_end|>\n',g:'<|im_start|>assistant\n',stop:['<|im_end|>']},
llama3:{n:'Llama 3',s:'<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>',u:'<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>',a:'<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>',g:'<|start_header_id|>assistant<|end_header_id|>\n\n',stop:['<|eot_id|>']},
mistral:{n:'Mistral / Mixtral',s:null,u:'[INST] {} [/INST]',a:' {}</s>',g:'',stop:['</s>','[INST]']},
gemma:{n:'Gemma',s:null,u:'<start_of_turn>user\n{}<end_of_turn>\n',a:'<start_of_turn>model\n{}<end_of_turn>\n',g:'<start_of_turn>model\n',stop:['<end_of_turn>']},
phi3:{n:'Phi-3',s:'<|system|>\n{}<|end|>\n',u:'<|user|>\n{}<|end|>\n',a:'<|assistant|>\n{}<|end|>\n',g:'<|assistant|>\n',stop:['<|end|>']},
alpaca:{n:'Alpaca',s:'{}\n\n',u:'### Instruction:\n{}\n\n',a:'### Response:\n{}\n\n',g:'### Response:\n',stop:['### Instruction:']},
raw:{n:'Plain text',s:'{}\n\n',u:'User: {}\n',a:'Assistant: {}\n',g:'Assistant:',stop:['\nUser:']}};
const P=(id,name,o)=>({id,name,mode:'openai',tpl:'chatml',temp:.7,top_p:.95,top_k:40,min_p:.05,rep:1.05,ctx:8192,sys:'',...o});
const PRESETS=()=>[P('balanced','Balanced',{}),
P('coding','Precise coding',{temp:.2,top_p:.9,top_k:20,rep:1.03,ctx:16384,sys:'Prefer correct, idiomatic, well-structured code.'}),
P('creative','Creative writing',{temp:1,top_p:.98,top_k:100,min_p:.02,rep:1.1}),
P('precise','Deterministic',{temp:.05,top_p:1,top_k:1,min_p:0,rep:1}),
P('reason','Reasoning (think first)',{temp:.6,top_k:20,min_p:0,ctx:16384,sys:'Think through the problem inside <think></think> tags first, then give the final answer.'}),
P('n-chatml','Native · ChatML / Qwen',{mode:'native',tpl:'chatml'}),P('n-llama3','Native · Llama 3',{mode:'native',tpl:'llama3'}),
P('n-mistral','Native · Mistral',{mode:'native',tpl:'mistral'}),P('n-gemma','Native · Gemma',{mode:'native',tpl:'gemma'})];
const STYLES={normal:['Normal','Balanced default',''],concise:['Concise','Short and direct','Be concise: short, direct answers without filler.'],explanatory:['Explanatory','Teach step by step','Explain clearly and thoroughly, step by step, with examples.'],formal:['Formal','Polished, professional','Use a formal, professional tone.'],creative:['Creative','Vivid and playful','Be imaginative, vivid and expressive.']};
const PERMS={ask:['Ask before edits','Approve every edit and command'],auto:['Edit automatically','Files change freely; commands ask'],plan:['Plan mode','Read-only: investigate and propose a plan'],yolo:['Bypass permissions','Run everything without asking']};

const defState=()=>({v:1,settings:{url:DEFAULT_URL||'http://localhost:5001',key:'',name:'',about:'',instr:'',style:'normal',theme:'dark',accent:'#d97757',font:'serif',preset:'balanced',effort:'medium',ceffort:'high',perm:'ask',skin:'classic',accentAll:false,anim:'full',crab:true,crabSize:64,crabx:{composer:true,avatar:true,roam:true,look:true},bg:{mode:'none',preset:'aurora',blur:0,dim:25,panel:75,v:0},rpc:{on:false,cid:'',img:'kobold',showModel:true,showProject:false}},presets:PRESETS(),chats:[],projects:[],sessions:[],macros:[],recentWs:[],ui:{tab:'chat',view:'home',chat:null,sess:null,proj:null,cpage:'profile'}});
let S=null,META={ws:'',sep:'/'};
const ST={files:[],tree:{},open:new Set(),seen:new Set(),panel:true,busy:false,conn:null};
const curPreset=()=>S.presets.find(p=>p.id===S.settings.preset)||S.presets[0];
const curEffort=k=>EFFORT.find(e=>e.id===S.settings[k==='code'?'ceffort':'effort'])||EFFORT[1];
const curChat=()=>S.ui.view==='chat'?S.chats.find(c=>c.id===S.ui.chat):null;
const curSess=()=>S.ui.view==='sess'?S.sessions.find(c=>c.id===S.ui.sess):null;
let saveT=0;const save=()=>{clearTimeout(saveT);saveT=setTimeout(()=>api('/api/state',S).catch(()=>{}),350)};

/* ============ markdown ============ */
const KW='def|class|return|if|else|elif|for|while|import|from|as|try|except|finally|with|lambda|yield|async|await|function|const|let|var|new|this|self|true|false|null|None|True|False|undefined|void|public|private|protected|static|struct|enum|interface|type|fn|pub|mut|use|impl|match|in|not|and|or|is|break|continue|switch|case|default|throw|raise|export|package|include|namespace|using|extends|implements|SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|JOIN|CREATE|TABLE';
const TOK=/("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|(\b\d[\d_.]*\b)|\b(KWS)\b|(\b[A-Za-z_]\w*(?=\())/.source.replace('KWS',KW);
const RE_HASH=new RegExp('(#[^\\n]*)|'+TOK,'g'),RE_SL=new RegExp('(\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/|<!--[\\s\\S]*?-->)|'+TOK,'g');
function hl(code,lang){lang=(lang||'').toLowerCase();const re=/^(py|python|sh|bash|shell|zsh|yaml|yml|toml|ini|rb|ruby|r|ps1|powershell|dockerfile|makefile|conf|env)$/.test(lang)?RE_HASH:RE_SL;re.lastIndex=0;
let out='',last=0,m;while((m=re.exec(code))){out+=esc(code.slice(last,m.index));const c=m[1]!=null?'c':m[2]!=null?'s':m[3]!=null?'n':m[4]!=null?'k':'f';out+=`<span class="tk-${c}">${esc(m[0])}</span>`;last=m.index+m[0].length;if(!m[0])re.lastIndex++}
return out+esc(code.slice(last))}
function cblock(lang,code){code=code.replace(/\n$/,'');return `<div class="cb"><div class="cbh"><span>${esc(lang||'text')}</span><span>${S.ui.tab==='code'?`<button data-a="savews" data-l="${esc(lang)}">Save to workspace</button>`:''}<button data-a="savecode" data-l="${esc(lang)}">Download</button><button data-a="copycode">Copy</button></span></div><pre><code>${hl(code,lang)}</code></pre></div>`}
function inl(s){const codes=[];s=s.replace(/`([^`\n]+)`/g,(m,c)=>{codes.push(c);return '\u0002'+(codes.length-1)+'\u0002'});s=esc(s);
s=s.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/(^|[^\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])/g,'$1<i>$2</i>').replace(/(^|\W)_(?!\s)([^_\n]+?)_(?!\w)/g,'$1<i>$2</i>').replace(/~~(.+?)~~/g,'<s>$1</s>');
s=s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
return s.replace(/\u0002(\d+)\u0002/g,(m,k)=>'<code>'+esc(codes[k])+'</code>')}
function md(src){const store=[];src=src.replace(/```([\w+#.-]*)[^\n]*\n([\s\S]*?)(?:```|$)/g,(m,l,c)=>{store.push(cblock(l,c));return '\n\u0001'+(store.length-1)+'\u0001\n'});
const L=src.split('\n'),out=[];let i=0,m;const LI=/^(\s*)([-*+]|\d+[.)])\s+(.*)/;
while(i<L.length){const l=L[i];if(!l.trim()){i++;continue}
if((m=/^\u0001(\d+)\u0001$/.exec(l))){out.push(store[m[1]]);i++;continue}
if((m=/^(#{1,6})\s+(.*)/.exec(l))){out.push(`<h${m[1].length}>${inl(m[2])}</h${m[1].length}>`);i++;continue}
if(/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)){out.push('<hr>');i++;continue}
if(/^\s*>/.test(l)){const b=[];while(i<L.length&&/^\s*>/.test(L[i]))b.push(L[i++].replace(/^\s*>\s?/,''));out.push('<blockquote>'+inl(b.join('\n')).replace(/\n/g,'<br>')+'</blockquote>');continue}
if(/\|/.test(l)&&i+1<L.length&&/^(?=.*\|)\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(L[i+1])){const cells=r=>r.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
const al=cells(L[i+1]).map(c=>/^:-+:$/.test(c)?'center':/-:$/.test(c)?'right':'left'),head=cells(l),rows=[];i+=2;while(i<L.length&&L[i].trim()&&/\|/.test(L[i]))rows.push(cells(L[i++]));
out.push('<div class="tw"><table><thead><tr>'+head.map((c,k)=>`<th style="text-align:${al[k]||'left'}">${inl(c)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((c,k)=>`<td style="text-align:${al[k]||'left'}">${inl(c)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>');continue}
if(LI.test(l)){const ord=/^\s*\d/.test(l),start=parseInt(LI.exec(l)[2])||1,items=[];
while(i<L.length&&(m=LI.exec(L[i]))){items.push({d:Math.floor(m[1].replace(/\t/g,'  ').length/2),t:m[3]});i++;while(i<L.length&&/^\s{2,}\S/.test(L[i])&&!LI.test(L[i]))items[items.length-1].t+=' '+L[i++].trim()}
out.push((ord?`<ol start="${start}">`:'<ul>')+items.map(x=>`<li style="margin-left:${x.d*18}px">${inl(x.t)}</li>`).join('')+(ord?'</ol>':'</ul>'));continue}
const b=[];while(i<L.length&&L[i].trim()&&!/^(#{1,6}\s|\s*>|\u0001)/.test(L[i])&&!LI.test(L[i]))b.push(L[i++]);if(!b.length)b.push(L[i++]);
out.push('<p>'+inl(b.join('\n')).replace(/\n/g,'<br>')+'</p>')}
return out.join('')}
const fixThink=c=>{c=c||'';const i=c.indexOf('</think>');if(i<0)return c;const j=c.indexOf('<think>');return j<0||j>i?'<think>'+c:c};
function renderMd(raw){raw=fixThink(raw);const re=/<think>([\s\S]*?)(<\/think>|$)/g;let html='',last=0,m;
while((m=re.exec(raw||''))){html+=md(raw.slice(last,m.index));const open=!m[2];html+=`<details class="think" ${open?'open':''}><summary>${open?'Thinking…':'Thought process'}</summary><div>${esc(m[1].trim())}</div></details>`;last=m.index+m[0].length;if(open)break}
return html+md((raw||'').slice(last))}

/* ============ requests to KoboldCpp ============ */
const mergeAdj=m=>{const o=[];for(const x of m){if(o.length&&o[o.length-1].role===x.role)o[o.length-1].content+='\n\n'+x.content;else o.push({role:x.role,content:x.content})}return o};
function fit(m,budget){m=m.map(x=>({...x}));const tot=()=>m.reduce((a,x)=>a+est(x.content)+8,0);
for(let i=0;i<m.length-3&&tot()>budget;i++)if(m[i].content.length>700)m[i].content=m[i].content.slice(0,350)+'\n…[trimmed to save context]…\n'+m[i].content.slice(-250);
while(m.length>1&&tot()>budget)m.shift();while(m.length>1&&m[0].role!=='user')m.shift();return m}
function mkPrompt(T,sys,msgs){let p='',first=true;if(sys&&T.s)p+=T.s.replace('{}',()=>sys);
for(const m of msgs){let c=m.content;if(m.role==='user'){if(first&&sys&&!T.s)c=sys+'\n\n'+c;first=false;p+=T.u.replace('{}',()=>c)}else p+=T.a.replace('{}',()=>c)}return p+T.g}
function mkReq(sys,msgs,o={}){const p=curPreset(),e=curEffort(o.code?'code':'chat'),ctx=+p.ctx||8192;
const maxTok=Math.max(256,Math.min(e.max,Math.floor(ctx*.5)));msgs=fit(mergeAdj(msgs),ctx-maxTok-est(sys)-120);
const temp=Math.max(Math.min(p.temp,.1),+(p.temp+e.dt).toFixed(2)),stop=o.code?['</tool_result>','<tool_result']:[],gk='kc'+uid();
if(p.mode==='native'){const T=TPL[p.tpl]||TPL.chatml;return{mode:'native',gk,payload:{prompt:mkPrompt(T,sys,msgs),max_context_length:ctx,max_length:maxTok,temperature:temp,top_p:p.top_p,top_k:p.top_k,min_p:p.min_p,rep_pen:p.rep,stop_sequence:[...T.stop,...stop].slice(0,16),trim_stop:true,genkey:gk}}}
return{mode:'openai',gk,payload:{model:'koboldcpp',messages:[{role:'system',content:sys},...msgs],max_tokens:maxTok,temperature:temp,top_p:p.top_p,top_k:p.top_k,min_p:p.min_p,rep_pen:p.rep,repeat_penalty:p.rep,stop,genkey:gk}}}
function abortGen(){if(ST.ctl)ST.ctl.abort();if(ST.req)api('/api/abort',{url:S.settings.url,key:S.settings.key,genkey:ST.req.gk}).catch(()=>{})}
async function doStream(req,onTok){const ctl=new AbortController();ST.ctl=ctl;ST.req=req;
try{const r=await fetch('/api/generate',{method:'POST',headers:{'X-KC':TOKEN,'Content-Type':'application/json'},body:JSON.stringify({url:S.settings.url,key:S.settings.key,mode:req.mode,payload:req.payload}),signal:ctl.signal});
if(!r.ok)throw new Error('Local server error '+r.status);const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
while((i=buf.indexOf('\n\n'))>=0){const ev=buf.slice(0,i);buf=buf.slice(i+2);if(!ev.startsWith('data:'))continue;let j;try{j=JSON.parse(ev.slice(5))}catch{continue}
if(j.err)throw new Error(j.err);if(j.t&&onTok(j.t)===false){abortGen();return}}}}
catch(e){if(e.name==='AbortError')return;throw e}finally{ST.ctl=null}}

/* ============ shell: menus, modals ============ */
function menu(anchor,items,above){const m=$('#menu');m.innerHTML=items.map((it,i)=>it.sep?'<hr>':`<button class="mi ${it.danger?'dg':''}" data-i="${i}" ${it.col?`style="--c:${it.col}"`:''}>${it.icon?ic(it.icon):''}<span>${esc(it.label)}</span>${it.hint?`<small>${esc(it.hint)}</small>`:''}${it.on?ic('check','ck'):''}</button>`).join('');
m.className='on';m._items=items;const r=anchor.getBoundingClientRect();m.style.left=Math.max(8,Math.min(r.left,innerWidth-m.offsetWidth-8))+'px';
m.style.top=Math.max(8,above?r.top-m.offsetHeight-8:Math.min(r.bottom+6,innerHeight-m.offsetHeight-8))+'px'}
const closeMenu=()=>{$('#menu').className=''};
function modal(html,wide){const o=$('#overlay');o.innerHTML=`<div class="modal ${wide?'wide':''}">${html}</div>`;o.className='on'}
const closeModal=()=>{const o=$('#overlay');o.className='';o.innerHTML=''};
function askText(title,val='',ph=''){return new Promise(res=>{modal(`<h3>${esc(title)}</h3><input id="mt" value="${esc(val)}" placeholder="${esc(ph)}"><div class="mb"><button class="btn" data-a="mcancel">Cancel</button><button class="btn pr" data-a="mok">Save</button></div>`);
ST.mres=res;const i=$('#mt');i.focus();i.select();i.onkeydown=e=>{if(e.key==='Enter')ACT.mok()}})}

/* ============ sidebar ============ */
const connHtml=()=>{const c=ST.conn,ok=c&&c.ok;return `<button id="conn" class="conn ${ok?'ok':''}" data-a="goconn" title="${esc(ok?'Connected to '+S.settings.url:(c&&c.err)||'Not connected')}"><i></i><span>${ok?esc((c.model||'KoboldCpp').replace(/^koboldcpp\//,'')||'KoboldCpp'):'Not connected'}</span></button>`};
function listItem(it,kind){const u=S.ui,on=kind==='code'?(u.view==='sess'&&u.sess===it.id):(u.view==='chat'&&u.chat===it.id);
return `<div class="li ${on?'on':''}" data-a="${kind==='code'?'opensess':'openchat'}" data-id="${it.id}">${it.pinned?ic('pin','pn'):''}<span class="t">${esc(it.title)}</span><button class="ib mo" data-a="itemmenu" data-id="${it.id}" data-k="${kind}">${ic('more')}</button></div>`}
function sidebar(){const u=S.ui,code=u.tab==='code',kind=code?'code':'chat',items=[...(code?S.sessions:S.chats)].sort((a,b)=>b.updated-a.updated),pins=items.filter(x=>x.pinned),rec=items.filter(x=>!x.pinned);
return `<aside class="side"><div class="brand ${ST.busy?'busy':''}">${LOGO(22)}KoboldCode</div>
<div class="seg"><button data-a="tab" data-v="chat" class="${code?'':'on'}">${ic('chat')} Chat</button><button data-a="tab" data-v="code" class="${code?'on':''}">${ic('code')} Code</button></div>
<button class="nav" data-a="new">${ic('plus')}<span>New ${code?'session':'chat'}</span></button>
<button class="nav" data-a="search">${ic('search')}<span>Search</span><kbd>Ctrl K</kbd></button>
${code?'':`<button class="nav ${u.view==='projects'||u.view==='project'?'on':''}" data-a="projects">${ic('folder')}<span>Projects</span></button>`}
<button class="nav ${u.view==='custom'?'on':''}" data-a="custom">${ic('sliders')}<span>Customize</span></button>
<div class="list">${pins.length?'<div class="sh">Pinned</div>'+pins.map(x=>listItem(x,kind)).join(''):''}<div class="sh">${code?'Sessions':'Recents'}</div>${rec.map(x=>listItem(x,kind)).join('')||emptyHtml('Nothing here yet')}</div>
<div class="foot">${connHtml()}<button class="ib" data-a="custom" title="Settings">${ic('gear')}</button></div></aside>`}

/* ============ composer ============ */
const lvHtml=i=>`<span class="lv">${EFFORT.map((_,k)=>`<i class="${k<=i?'on':''}"></i>`).join('')}</span>`;
function composer(kind,ph){const c=kind==='code',s=c?curSess():null,p=curPreset(),e=curEffort(kind),perm=s?s.perm:S.settings.perm;
return `<div class="comp" data-ef="${e.id}" style="--ef:${e.col};--efc:${e.on}">${crabHtml()}<div id="files">${filesHtml()}</div><textarea id="inp" rows="1" placeholder="${esc(ph)}"></textarea><div class="crow">
<button class="ib" data-a="attach" title="Attach text files">${ic('clip')}</button>
${c?`<button class="pill" data-a="permmenu">${ic('shield')}<span class="pt">${PERMS[perm][0]}</span>${ic('chev')}</button>`:''}<span class="sp"></span>
<button class="pill" data-a="presetmenu" title="Sampler preset"><span class="pt">${esc(p.name)}</span>${ic('chev')}</button>
<button class="pill ef" data-a="effortmenu" title="Effort">${ic('bolt')}<span class="pt">${e.name}</span>${lvHtml(EFFORT.indexOf(e))}${ic('chev')}</button>
${ST.busy?`<button class="send stop" data-a="stop" title="Stop (Esc)">${ic('stop')}</button>`:`<button class="send" data-a="send" title="Send (Enter)">${ic('up')}</button>`}</div></div>`}
const filesHtml=()=>ST.files.map((f,i)=>`<span class="fc">${ic('file')}${esc(f.name)}<button data-a="rmfile" data-i="${i}">${ic('x')}</button></span>`).join('');
const autosize=t=>{t.style.height='auto';t.style.height=Math.min(t.scrollHeight,240)+'px'};

/* ============ chat views ============ */
function topbar(inner){return `<header class="top"><button class="ib sidebtn" data-a="togside">${ic('menu')}</button>${inner}</header>`}
const nw=id=>ST.seen.has(id)?'':' new';
const dayLine=()=>new Date().toLocaleDateString(undefined,{weekday:'long'})+' · '+new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
function viewHome(){const b=bucket(),nm=S.settings.name?', '+S.settings.name:'';ST.greet??=pick(GREET[b]).replace('{n}',nm);
return topbar('<span class="sp"></span>')+`<div class="home" style="--glow:${GLOW[b]}"><div class="hello">${LOGO(40)}<h1>${esc(ST.greet)}</h1></div><div class="sub" id="clock">${dayLine()}</div>${composer('chat','How can I help you today?')}
<div class="chips">${CHIPS[b].map(c=>`<button class="chip" data-a="chip" data-t="${esc(c[1])}">${esc(c[0])}</button>`).join('')}</div></div>`}
function userHtml(m){return `<div class="m u${nw(m.id)}" data-mid="${m.id}">${m.files?.length?`<div class="fchips">${m.files.map(f=>`<span class="fc">${ic('file')}${esc(f)}</span>`).join('')}</div>`:''}${(m.text??m.content)?`<div class="ub">${esc(m.text??m.content)}</div>`:''}</div>`}
const asstHtml=m=>(m.content?renderMd(m.content):(m.streaming?'<span class="dots"><i></i><i></i><i></i></span>':''))+(m.streaming&&m.content?'<span class="cur"></span>':'');
function msgHtml(m,last,prev){if(m.role==='user')return userHtml(m);
return `<div class="m a${avOn()?' av':''} ${last?'last':''}${nw(m.id)}" data-mid="${m.id}">${avHtml(m,prev)}<div class="ab">${asstHtml(m)}</div>${m.streaming?'':`<div class="acts"><button class="ib" data-a="copymsg" data-id="${m.id}" title="Copy">${ic('copy')}</button>${last?`<button class="ib" data-a="regen" title="Regenerate">${ic('refresh')}</button>`:''}</div>`}</div>`}
function viewChat(c){const proj=c.projectId&&S.projects.find(p=>p.id===c.projectId);
return topbar(`<div class="ttl"><span>${proj?`<span class="crumb" data-a="openproject" data-id="${proj.id}">${esc(proj.name)}</span> / `:''}<b>${esc(c.title)}</b></span></div><span class="sp"></span><span class="mu" style="font-size:12px">${esc(curPreset().name)}</span>`)+
`<div class="msgs" id="msgs"><div class="col">${c.msgs.map((m,i)=>msgHtml(m,i===c.msgs.length-1,c.msgs[i-1])).join('')}</div></div><div class="dock"><div class="col">${composer('chat','Reply…')}<div class="fine">Running locally via KoboldCpp · answers can be wrong</div></div></div>`}
function sysChat(c){const s=S.settings,p=curPreset(),e=curEffort('chat'),b=bucket();
let t=`You are a helpful, thoughtful AI assistant running locally through KoboldCpp. Today is ${new Date().toLocaleDateString(undefined,{weekday:'long',year:'numeric',month:'long',day:'numeric'})}; it is ${b==='night'?'late at night':b==='day'?'the afternoon':b} for the user.`;
if(s.name)t+=` The user's name is ${s.name}.`;if(s.about)t+=`\nAbout the user: ${s.about}`;if(STYLES[s.style][2])t+=`\nStyle: ${STYLES[s.style][2]}`;if(s.instr)t+=`\nUser instructions: ${s.instr}`;
const pr=c.projectId&&S.projects.find(x=>x.id===c.projectId);if(pr){t+=`\n\n# Project: ${pr.name}\n${pr.instr||''}`;let n=0;for(const f of pr.files||[]){if(n>14000)break;const x=f.text.slice(0,14000-n);n+=x.length;t+=`\n\n## File: ${f.name}\n${x}`}}
if(p.sys)t+='\n'+p.sys;return t+`\nEffort: ${e.hint}\nUse Markdown; put code in fenced blocks with a language tag.`}
async function runChat(c){const a={id:uid(),role:'assistant',content:'',streaming:true};c.msgs.push(a);ST.busy=true;ST.stopReq=false;ST.run=c.id;c.updated=Date.now();save();render();setCrab('think');
try{await doStream(mkReq(sysChat(c),c.msgs.slice(0,-1).filter(m=>!m.kind).map(m=>({role:m.role,content:m.content}))),t=>{a.content+=t;crabFor(a.content);paint(a)})}
catch(err){a.content+=(a.content?'\n\n':'')+'**Error:** '+err.message+'\n\nCheck that KoboldCpp is running and the URL in *Customize → Connection* is right.'}
a.streaming=false;ST.busy=false;c.updated=Date.now();save();render();
if(ST.stopReq)setCrab('idle','');else if(a.content.includes('**Error:**'))setCrab('sad','Is KoboldCpp running?',5200);else setCrab('happy',undefined,3600)}
function chatSend(text,files){let c=curChat();if(!c){c={id:uid(),title:titleFrom(text||files[0]?.name),msgs:[],projectId:S.ui.view==='project'?S.ui.proj:null,pinned:false,created:Date.now(),updated:Date.now()};S.chats.push(c)}
S.ui.chat=c.id;S.ui.view='chat';const body=files.map(f=>`<attachment name="${f.name}">\n${f.text}\n</attachment>`).join('\n')+(files.length?'\n\n':'')+text;
c.msgs.push({id:uid(),role:'user',content:body,text,files:files.map(f=>f.name)});runChat(c)}

/* streaming paint (only touches the one message) */
function paint(m){ST.dirty=m;if(ST.raf)return;ST.raf=requestAnimationFrame(()=>{ST.raf=0;const x=ST.dirty,t=performance.now();if(t-(ST.lp||0)<40){paint(x);return}ST.lp=t;
const root=$(`[data-mid="${x.id}"]`);if(!root)return;const el=$('.ab',root),b=$('#msgs'),code=S.ui.tab==='code',near=b&&b.scrollHeight-b.scrollTop-b.clientHeight<140;
const tmp=document.createElement('div');tmp.innerHTML=code?codeAsst(x):asstHtml(x);morph(el,tmp);const mini=$('.avt .pinch',root);if(mini)setMini(mini,moodFor(x.content,code));
if(near&&b)b.scrollTop=b.scrollHeight;const cm=$('#ctxm');if(cm)cm.outerHTML=ctxMeter(curSess())})}

/* ============ projects ============ */
function viewProjects(){return topbar('<b>Projects</b><span class="sp"></span><button class="btn pr" data-a="newproj">'+ic('plus')+' New project</button>')+
`<div class="page"><div class="pg">${S.projects.map(p=>`<div class="pcard" data-a="openproject" data-id="${p.id}"><b>${esc(p.name)}</b><p>${esc(p.desc||'No description')}</p><small>${S.chats.filter(c=>c.projectId===p.id).length} chats · ${(p.files||[]).length} files</small></div>`).join('')||'<div class="empty" style="grid-column:1/-1;text-align:center;padding:60px 0">Projects bundle instructions and reference files for related chats.<br><br><button class="btn pr" data-a="newproj">Create your first project</button></div>'}</div></div>`}
function viewProject(p){const cs=S.chats.filter(c=>c.projectId===p.id).sort((a,b)=>b.updated-a.updated);
return topbar(`<span class="crumb" data-a="projects">Projects</span><span class="mu">/</span><b>${esc(p.name)}</b><span class="sp"></span><button class="btn dg" data-a="delproj" data-id="${p.id}">Delete</button>`)+
`<div class="page"><div class="two"><div><h2>${esc(p.name)}</h2><p class="mu">${esc(p.desc||'')}</p>${composer('chat','Start a chat in this project…')}
<div class="pchats">${cs.map(c=>`<div class="li" data-a="openchat" data-id="${c.id}"><span class="t">${esc(c.title)}</span><small class="mu">${new Date(c.updated).toLocaleDateString()}</small></div>`).join('')||emptyHtml('No chats yet')}</div></div>
<div><div class="card"><h4>Instructions<button class="ib" data-a="editproj" data-id="${p.id}">${ic('edit')}</button></h4><p>${esc(p.instr||'Add instructions that apply to every chat in this project.')}</p></div><br>
<div class="card"><h4>Files<button class="ib" data-a="addpfile" data-id="${p.id}">${ic('plus')}</button></h4>${(p.files||[]).map((f,i)=>`<div class="row">${ic('file')}<span style="flex:1;overflow:hidden;text-overflow:ellipsis">${esc(f.name)}</span><button class="ib" data-a="rmpfile" data-id="${p.id}" data-i="${i}">${ic('x')}</button></div>`).join('')||'<p>Reference files the model can read in every chat.</p>'}</div></div></div></div>`}
function projModal(p){modal(`<h3>${p?'Edit project':'New project'}</h3><label>Name</label><input id="pn" value="${esc(p?.name||'')}"><label>Description</label><input id="pd" value="${esc(p?.desc||'')}"><label>Instructions</label><textarea id="pi" rows="6" placeholder="How should the model behave in this project?">${esc(p?.instr||'')}</textarea><div class="mb"><button class="btn" data-a="mcancel">Cancel</button><button class="btn pr" data-a="saveproj" data-id="${p?.id||''}">Save</button></div>`)}

/* ============ customize ============ */
const fld=(l,h,sm)=>`<div class="f"><label>${l}</label>${h}${sm?`<small>${sm}</small>`:''}</div>`;
const bi=(p,type='text',x='')=>`<input type="${type}" data-bind="${p}" value="${esc(getPath(S,p))}" ${x}>`;
const ta=(p,r=4,ph='')=>`<textarea rows="${r}" data-bind="${p}" placeholder="${esc(ph)}">${esc(getPath(S,p))}</textarea>`;
const sel=(p,o,rr)=>`<select data-bind="${p}" ${rr?'data-rr="1"':''}>${o.map(([v,l])=>`<option value="${v}" ${String(getPath(S,p))===String(v)?'selected':''}>${esc(l)}</option>`).join('')}</select>`;
const rng=(p,a,b,st)=>`<div class="rng"><input type="range" data-bind="${p}" min="${a}" max="${b}" step="${st}" value="${getPath(S,p)}"><output>${getPath(S,p)}</output></div>`;
const PAGES={
profile:()=>`<h2>Profile & style</h2><p class="mu">Applied to every chat.</p>${fld('Your name',bi('settings.name'),'Used in the time-of-day greeting.')}${fld('About you',ta('settings.about',3,'Your work, interests, what to keep in mind…'))}${fld('Custom instructions',ta('settings.instr',5,'e.g. Always answer in English. Prefer TypeScript.'))}${fld('Response style',sel('settings.style',Object.entries(STYLES).map(([k,v])=>[k,v[0]+' — '+v[1]])))}`,
look:()=>`<h2>Appearance</h2>${fld('Theme',sel('settings.theme',[['dark','Dark'],['light','Light'],['auto','Auto — light by day, dark by night']],1))}${fld('Accent',`<div class="sw">${['#d97757','#e0a14a','#6aa86b','#4f9bd9','#9b7be0','#d9608f'].map(c=>`<button data-a="accent" data-c="${c}" class="${S.settings.accent===c?'on':''}" style="background:${c}"></button>`).join('')}</div>`)}${fld('Response font',sel('settings.font',[['serif','Serif (Claude-like)'],['sans','Sans-serif']],1))}`,
conn:()=>`<h2>Connection</h2><p class="mu">KoboldCpp (or any KoboldAI/OpenAI-compatible server).</p>${fld('Server URL',bi('settings.url'),'Default KoboldCpp port is 5001.')}${fld('API key (optional)',bi('settings.key','password'),'Only if you started KoboldCpp with --password.')}<button class="btn pr" data-a="testconn">Test connection</button><div id="testres" style="margin-top:12px"></div>
<div class="card" style="margin-top:20px"><b>Start KoboldCpp</b><p style="margin-top:6px;font:12.5px var(--mono)">koboldcpp --model your-model.gguf --port 5001 --contextsize 16384</p></div>`,
presets:()=>{const p=curPreset(),i=S.presets.indexOf(p),b=k=>`presets.${i}.${k}`;
return `<h2>Presets & sampling</h2><p class="mu">A preset bundles API mode, prompt template and samplers. Switch from the composer anywhere.</p>
<div class="pgrid">${S.presets.map(x=>`<button class="pc ${x===p?'on':''}" data-a="usepreset" data-id="${x.id}"><b>${esc(x.name)}</b><small>${x.mode==='native'?'native · '+TPL[x.tpl].n:'OpenAI chat · T '+x.temp}</small></button>`).join('')}</div>
<div class="card">${fld('Name',bi(b('name')))}${fld('API mode',sel(b('mode'),[['openai','OpenAI-compatible chat (KoboldCpp applies the model template)'],['native','Native KoboldAI stream (we apply the template below)']],1))}
${p.mode==='native'?fld('Prompt template',sel(b('tpl'),Object.entries(TPL).map(([k,v])=>[k,v.n])),'Match this to your model family.'):''}${fld('Context size',bi(b('ctx'),'number','min="512" step="512"'),'Keep ≤ the --contextsize KoboldCpp was started with.')}
${fld('Temperature',rng(b('temp'),0,2,.05))}${fld('Top-p',rng(b('top_p'),0,1,.01))}${fld('Top-k',rng(b('top_k'),0,200,1))}${fld('Min-p',rng(b('min_p'),0,1,.01))}${fld('Repetition penalty',rng(b('rep'),1,1.5,.01))}${fld('Extra system prompt',ta(b('sys'),3))}</div>
<div class="mb" style="justify-content:flex-start"><button class="btn" data-a="duppreset">Duplicate</button><button class="btn dg" data-a="delpreset">Delete</button><button class="btn" data-a="resetpresets">Reset all</button></div>`},
code:()=>`<h2>Code agent</h2>${fld('Default permission mode',sel('settings.perm',Object.entries(PERMS).map(([k,v])=>[k,v[0]+' — '+v[1]])))}
<div class="f"><label>Custom slash commands</label><small>Type /name in the Code tab. Use $ARGS for text typed after the command.</small>${S.macros.map((m,i)=>`<div class="mrow"><input data-bind="macros.${i}.n" value="${esc(m.n)}" placeholder="name"><textarea rows="2" data-bind="macros.${i}.p" placeholder="prompt">${esc(m.p)}</textarea><button class="ib" data-a="delmacro" data-i="${i}">${ic('x')}</button></div>`).join('')}<button class="btn" data-a="addmacro" style="margin-top:8px">${ic('plus')} Add command</button></div>`,
data:()=>`<h2>Data</h2><p class="mu">Stored in <code>~/.koboldcode/state.json</code> on this machine only.</p><div class="mb" style="justify-content:flex-start"><button class="btn" data-a="exportall">${ic('dl')} Export everything</button><button class="btn" data-a="importall">Import…</button><button class="btn dg" data-a="clearchats">Delete all chats</button></div>`};
function viewCustom(){const pg=S.ui.cpage||'profile',nav=[['profile','Profile & style'],['look','Appearance'],['conn','Connection'],['presets','Presets & sampling'],['code','Code agent'],['discord','Discord'],['data','Data']];
return topbar('<b>Customize</b>')+`<div class="cust"><nav class="cn">${nav.map(([k,l])=>`<button data-a="cpage" data-v="${k}" class="${pg===k?'on':''}">${l}</button>`).join('')}</nav><div class="cb2">${PAGES[pg]()}</div></div>`}

/* ============ search ============ */
function openSearch(){modal(`<div class="sbox">${ic('search')}<input id="sq" placeholder="Search chats, sessions and projects…"></div><div class="sres" id="sres"></div>`,1);
const f=()=>{const q=$('#sq').value.toLowerCase().trim(),res=[];
const scan=(arr,k)=>arr.forEach(x=>{const body=x.msgs.map(m=>m.text||m.content||'').join('\n'),i=q?body.toLowerCase().indexOf(q):-1;if(!q||x.title.toLowerCase().includes(q)||i>=0)res.push({x,k,snip:i>=0?body.slice(Math.max(0,i-35),i+90).replace(/\s+/g,' '):''})});
scan(S.chats,'chat');scan(S.sessions,'code');S.projects.forEach(p=>{if(!q||(p.name+p.desc).toLowerCase().includes(q))res.push({x:{...p,updated:p.created||0,title:p.name},k:'proj',snip:p.desc||''})});
res.sort((a,b)=>b.x.updated-a.x.updated);const act={chat:'openchat',code:'opensess',proj:'openproject'},icn={chat:'chat',code:'code',proj:'folder'};
$('#sres').innerHTML=res.slice(0,30).map(r=>`<button class="sr" data-a="${act[r.k]}" data-id="${r.x.id}">${ic(icn[r.k])}<span><b>${esc(r.x.title)}</b><small>${esc(r.snip)}</small></span></button>`).join('')||'<div class="empty">No matches</div>'};
$('#sq').oninput=f;f();$('#sq').focus()}

const SPRITE={"pal": {"o": "#e8764a", "d": "#b9532c", "l": "#f6a27a", "k": "#1c1410", "w": "#ffffff", "p": "#ff9aa8", "s": "#00000040"}, "shadow": [["s", 4, 12, 12, 1]], "stalks": [["o", 6, 1, 1, 2], ["o", 13, 1, 1, 2], ["k", 6, 0, 1, 1], ["k", 13, 0, 1, 1]], "body": [["o", 6, 3, 8, 1], ["o", 4, 4, 12, 5], ["d", 5, 9, 10, 1], ["l", 4, 4, 1, 2]], "claw": [["o", 0, 3, 1, 1], ["o", 2, 3, 1, 1], ["o", 0, 4, 3, 2], ["d", 0, 5, 3, 1], ["o", 2, 6, 1, 1], ["o", 3, 7, 1, 1]], "legsA": [["d", 5, 10, 1, 2], ["d", 14, 10, 1, 2]], "legsB": [["d", 7, 10, 1, 2], ["d", 12, 10, 1, 2]], "cheeks": [["p", 5, 7, 1, 1], ["p", 14, 7, 1, 1]], "eyes": {"normal": [["k", 6, 4, 2, 2], ["k", 12, 4, 2, 2], ["w", 6, 4, 1, 1], ["w", 12, 4, 1, 1]], "happy": [["k", 6, 5, 1, 1], ["k", 7, 4, 1, 1], ["k", 8, 5, 1, 1], ["k", 11, 5, 1, 1], ["k", 12, 4, 1, 1], ["k", 13, 5, 1, 1]], "dead": [["k", 6, 4, 1, 1], ["k", 8, 4, 1, 1], ["k", 7, 5, 1, 1], ["k", 6, 6, 1, 1], ["k", 8, 6, 1, 1], ["k", 11, 4, 1, 1], ["k", 13, 4, 1, 1], ["k", 12, 5, 1, 1], ["k", 11, 6, 1, 1], ["k", 13, 6, 1, 1]], "sleep": [["k", 6, 5, 3, 1], ["k", 11, 5, 3, 1]]}, "mouth": {"normal": [["k", 9, 7, 2, 1]], "happy": [["k", 8, 7, 1, 1], ["k", 9, 8, 2, 1], ["k", 11, 7, 1, 1]], "sad": [["k", 8, 8, 4, 1]], "sleep": [["k", 9, 7, 2, 1]]}};
/* ============ skins + wallpaper ============ */
const SKINS=[
{id:'classic',n:'Classic',g:'Colors',sw:['#262624','#1f1e1d','#d97757']},
{id:'midnight',n:'Midnight',g:'Colors',base:'dark',sw:['#12141f','#0d0f18','#7aa2f7']},
{id:'nord',n:'Nord',g:'Colors',base:'dark',sw:['#2e3440','#272c36','#88c0d0']},
{id:'dracula',n:'Dracula',g:'Colors',base:'dark',sw:['#282a36','#21222c','#bd93f9']},
{id:'solarized',n:'Solarized',g:'Colors',base:'dark',sw:['#002b36','#00212b','#b58900']},
{id:'rose',n:'Rosé',g:'Colors',base:'dark',sw:['#191724','#13111c','#eb6f92']},
{id:'forest',n:'Forest',g:'Colors',base:'dark',sw:['#1a211b','#141a15','#8fbf7f']},
{id:'terminal',n:'Terminal',g:'Colors',base:'dark',own:1,sw:['#050805','#030503','#4dff7a']},
{id:'glass',n:'Liquid Glass',g:'Glass',base:'dark',own:1,sw:['#4f46e5','#06b6d4','#ec4899']},
{id:'macos',n:'macOS',g:'Operating systems',base:'light',own:1,sw:['#f6f6f8','#e9e9ee','#0a84ff']},
{id:'win11',n:'Windows 11',g:'Operating systems',base:'light',own:1,sw:['#f3f3f3','#eaeaea','#0067c0']},
{id:'vista',n:'Windows Vista',g:'Operating systems',base:'light',own:1,sw:['#2d6fb0','#6d9bd0','#2a7bd8']},
{id:'winxp',n:'Windows XP',g:'Operating systems',base:'light',own:1,sw:['#2a62d0','#ece9d8','#4aa02c']},
{id:'win98',n:'Windows 98',g:'Operating systems',base:'light',own:1,sw:['#008080','#c0c0c0','#000080']}];
const SKIN_BG={glass:'aurora',vista:'vista'};
const TIMEBG={night:'linear-gradient(160deg,#080c1f,#1a2352 55%,#3c2a66)',dawn:'linear-gradient(160deg,#ffb199,#ff7e8b 45%,#6a5a9a)',morning:'linear-gradient(160deg,#ffe29f,#ffa99f 55%,#7aa7e0)',day:'linear-gradient(160deg,#6dd5ed,#2193b0 60%,#1b5e86)',evening:'linear-gradient(160deg,#ff7e5f,#c8508a 55%,#2d2a6e)'};
const WALL={aurora:'radial-gradient(ellipse 60% 50% at 20% 20%,#4f46e5cc,transparent 62%),radial-gradient(ellipse 55% 45% at 82% 30%,#06b6d4bb,transparent 62%),radial-gradient(ellipse 60% 50% at 50% 95%,#ec4899bb,transparent 62%),#0b1020',
sunset:'linear-gradient(135deg,#ff7e5f,#feb47b 40%,#8e44ad)',ocean:'linear-gradient(160deg,#0f2027,#203a43 50%,#2c5364)',forest:'linear-gradient(160deg,#0b1f14,#1d4a2e 55%,#3c7a4a)',graphite:'linear-gradient(160deg,#1c1c1e,#2c2c2e 60%,#3a3a3c)',
vista:'radial-gradient(ellipse 60% 45% at 18% 85%,#3fbf74aa,transparent 60%),radial-gradient(ellipse 60% 50% at 85% 18%,#59b6f0cc,transparent 60%),linear-gradient(135deg,#0d5a3c,#1a8a6a 42%,#2d6fb0)'};
const wallCss=p=>p==='time'?TIMEBG[bucket()]:(WALL[p]||WALL.aurora);
const WALLS=[['none','None'],['time','Time of day'],['aurora','Aurora'],['sunset','Sunset'],['ocean','Ocean'],['forest','Forest'],['graphite','Graphite']];
function applyTheme(){const s=S.settings,d=document.documentElement,b=document.body,sk=SKINS.find(x=>x.id===s.skin)||SKINS[0],h=new Date().getHours();
const base=sk.base||(s.theme==='auto'?(h>=7&&h<18?'light':'dark'):s.theme);d.dataset.theme=base;d.dataset.skin=sk.id;
const setp=(k,v)=>v==null?d.style.removeProperty(k):d.style.setProperty(k,v);
setp('--ac',sk.id==='classic'||s.accentAll?s.accent:null);setp('--ft',sk.own?null:(s.font==='serif'?'var(--serif)':'var(--sans)'));
const bg=s.bg||{mode:'none'};let eff=null;
if(bg.mode==='image'&&bg.v)eff={img:1,...bg};else if(bg.mode==='preset')eff=bg;else if(SKIN_BG[sk.id])eff={mode:'preset',preset:SKIN_BG[sk.id],blur:0,dim:0,panel:75};
b.classList.toggle('wp',!!eff);b.classList.toggle('gl',sk.id==='glass');b.classList.toggle('avon',avOn());b.classList.toggle('crab',!!s.crab&&!(s.crabx&&s.crabx.composer===false));
d.classList.toggle('anim-off',s.anim==='off');d.classList.toggle('anim-sub',s.anim==='subtle');
const L=$('#bgl');if(!L)return;
if(eff){const blur=+eff.blur||0;L.style.background=eff.img?`url(/api/wallpaper?t=${TOKEN}&v=${eff.v}) center/cover`:wallCss(eff.preset);L.style.filter=blur?`blur(${blur}px)`:'none';L.style.transform=blur?'scale(1.08)':'none';
L.style.setProperty('--dim',(+eff.dim||0)/100);L.style.setProperty('--dimc',base==='light'?'#fff':'#000');b.style.setProperty('--pn',(eff.panel??75)+'%');L.classList.toggle('aur',!eff.img&&eff.preset==='aurora')}}
function pickWall(file){return new Promise((res,rej)=>{const img=new Image(),u=URL.createObjectURL(file);img.onload=()=>{const k=Math.min(1,2200/Math.max(img.width,img.height)),c=document.createElement('canvas');c.width=Math.round(img.width*k);c.height=Math.round(img.height*k);
c.getContext('2d').drawImage(img,0,0,c.width,c.height);URL.revokeObjectURL(u);res(c.toDataURL('image/jpeg',.86))};img.onerror=()=>rej(new Error('not an image'));img.src=u})}
document.addEventListener('change',async e=>{if(e.target.id!=='bgf')return;const f=e.target.files[0];e.target.value='';if(!f)return;
try{const data=await pickWall(f),r=await api('/api/wallpaper',{data});if(!r.ok)throw new Error(r.error);Object.assign(S.settings.bg,{mode:'image',v:r.v});save();render();toast('Wallpaper set')}catch(err){toast('Could not use that image: '+err.message)}});

/* ============ Pinch, the pixel crab ============ */
const CRAB={mood:'idle',msg:'',t:Date.now(),x:.93,ex:0,ey:0};
const LINES={think:['Thinking…','Hmm…','Let me see…'],write:['Typing away!','Writing code…','Pinching bits…'],run:['Running it!','Scuttling…'],look:['Reading files…','Looking around…'],ask:['May I?','Your call!'],happy:['Done!','Nailed it!','Shipped it!'],sad:['Oops…','That broke…'],sleep:['zzz…']};
const QUIPS={night:['Late-night coding hits different.','Sleep… after this commit.'],dawn:['Early bird gets the bug.','Coffee first?'],morning:['Good morning! Ship something.','Fresh day, fresh repo.'],day:['Have you committed lately?','Tests first?'],evening:['Golden hour = golden code.','One more feature?']};
const HELLO={night:"Late-night jam? I'm in.",dawn:'Up early! Coffee?',morning:'Morning! What are we building?',day:'Afternoon! Ready when you are.',evening:'Evening! Let us ship something.'};
const EFCOL=Object.fromEntries(EFFORT.map(e=>[e.id,e.col]));
const cx=()=>(S&&S.settings.crabx)||{};
const rc=r=>`<rect x="${r[1]}" y="${r[2]}" width="${r[3]}" height="${r[4]}" fill="${SPRITE.pal[r[0]]}"/>`;
const mirr=r=>[r[0],20-r[1]-r[3],r[2],r[3],r[4]];
const SHADES='<rect x="5" y="4" width="4" height="2" fill="#15151a"/><rect x="11" y="4" width="4" height="2" fill="#15151a"/><rect x="9" y="4" width="2" height="1" fill="#15151a"/><rect x="5" y="4" width="1" height="1" fill="#6ea8fe"/><rect x="11" y="4" width="1" height="1" fill="#6ea8fe"/>';
/* outfit grows with the effort level: scarf → headband → sparkles → shades + rainbow scarf */
function outfit(ef){const c=EFCOL[ef]||'#6fcf97',i=Math.max(0,EFFORT.findIndex(e=>e.id===ef));let x=`<rect${i===4?' class="rb"':''} x="5" y="9" width="10" height="1" fill="${c}"/>`;
if(i>=2)x+=`<rect x="6" y="3" width="8" height="1" fill="${c}"/>`;if(i>=3)x+=`<rect class="tw" x="2" y="1" width="1" height="1" fill="${c}"/><rect class="tw tw2" x="17" y="2" width="1" height="1" fill="${c}"/>`;return x}
function crabSvg(mood,ef){ef=ef||curEffort(S.ui.tab==='code'?'code':'chat').id;const E={happy:'happy',sad:'dead',sleep:'sleep'}[mood]||'normal',M={happy:'happy',sad:'sad',sleep:'sleep'}[mood]||'normal',cl=SPRITE.claw;let x='';
if(mood==='happy')x=[[3,-3,'#f2c94c'],[16,-4,'#6ea8fe'],[8,-5,'#e056c8'],[12,-4,'#6fcf97'],[1,-2,'#f2994a'],[18,-2,'#f2c94c']].map(([cx2,dy,c],i)=>`<rect class="cf" x="${cx2}" y="3" width="1" height="1" fill="${c}" style="--dx:${(i%2?1:-1)*(2+i)}px;--dy:${dy-3}px;animation-delay:${i*.08}s"/>`).join('');
if(mood==='sleep')x='<text class="zz" x="16" y="3">z</text><text class="zz" x="18" y="1" style="animation-delay:1.2s">z</text>';
if(mood==='sad')x='<rect class="drop" x="15" y="3" width="1" height="2" fill="#7fc4ff"/>';
if(mood==='ask')x='<text x="9" y="-1" font-size="5" font-weight="700" font-family="monospace" fill="currentColor">?</text>';
const sh=ef==='ultracode'&&!['happy','sad','sleep'].includes(mood)?SHADES:'',R=a=>a.map(rc).join('');
return `<svg viewBox="0 0 20 13" shape-rendering="crispEdges">${R(SPRITE.shadow)}<g class="cr-all"><g class="cr-lgA">${R(SPRITE.legsA)}</g><g class="cr-lgB">${R(SPRITE.legsB)}</g>${R(SPRITE.stalks)}${R(SPRITE.body)}${mood==='happy'||mood==='sleep'?R(SPRITE.cheeks):''}${outfit(ef)}<g class="cr-look"><g class="cr-eyes">${R(SPRITE.eyes[E])}</g></g>${sh}${R(SPRITE.mouth[M])}
<g class="cr-cl cr-cl-l">${R(cl)}</g><g class="cr-cl cr-cl-r">${cl.map(r=>rc(mirr(r))).join('')}</g></g>${x}</svg>`}
const miniCrab=(mood='idle',px=30)=>`<span class="pinch mini m-${mood}" data-mood="${mood}" style="--cs:${px}px">${crabSvg(mood)}</span>`;
function setMini(el,mood){if(el.dataset.mood===mood)return;el.dataset.mood=mood;el.className='pinch mini m-'+mood;el.innerHTML=crabSvg(mood)}
const emptyHtml=t=>`<div class="empty pe">${S&&S.settings.crab?miniCrab('sleep',30):''}<span>${t}</span></div>`;
const crabClass=()=>`pinch m-${CRAB.mood}${CRAB.x<.35?' pbr':''}`;
const crabInner=()=>(CRAB.msg?`<div class="pb">${esc(CRAB.msg)}</div>`:'')+crabSvg(CRAB.mood);
const crabHtml=()=>S&&S.settings.crab&&cx().composer?`<div id="crab" class="${crabClass()}" data-a="crabpoke" style="--cs:${S.settings.crabSize||64}px;--px:${CRAB.x};--ex:${CRAB.ex};--ey:${CRAB.ey}" title="Pinch">${crabInner()}</div>`:'';
function paintCrab(){const el=$('#crab');if(!el)return;el.className=crabClass();el.innerHTML=crabInner()}
function crabGo(x,sec){CRAB.x=x;const el=$('#crab');if(el){el.style.setProperty('--wd',sec+'s');el.style.setProperty('--px',x)}}
function setCrab(mood,msg,hold,quiet){if(!S||!S.settings.crab)return;clearTimeout(CRAB.rt);CRAB.mood=mood;if(!quiet)CRAB.t=Date.now();CRAB.msg=msg===undefined?(LINES[mood]?pick(LINES[mood]):''):(msg||'');
if(['think','write','run','look','ask'].includes(mood)&&CRAB.x!==.93)crabGo(.93,.8);paintCrab();if(hold)CRAB.rt=setTimeout(()=>{CRAB.mood='idle';CRAB.msg='';paintCrab()},hold)}
const TOOLMOOD={write_file:'write',edit_file:'write',delete_path:'write',run_command:'run',read_file:'look',list_dir:'look',search:'look'};
const crabThinking=c=>/<think>(?![\s\S]*<\/think>)/.test(c||'');
function moodFor(c,code){if(code){const pt=parseTool(c);if(pt)return TOOLMOOD[pt.name]||'think'}return crabThinking(c)||!(c||'').trim()?'think':'write'}
function crabFor(c){const m=moodFor(c,false);if(CRAB.mood!==m)setCrab(m,m==='think'?'Thinking…':'')}
function crabForTool(pt,c){const m=moodFor(c,true);if(CRAB.mood!==m)setCrab(m)}
function crabAfter(t,r){if(!r.ok)setCrab('sad','That failed…',2200);else if(t.name==='run_command'&&/\bOK\b|passed|0 failed|All tests/i.test(r.out||''))setCrab('happy','Tests passed!',2400);else setCrab('think')}
const crabWake=()=>{CRAB.t=Date.now();if(CRAB.mood==='sleep')setCrab('idle','')};
['keydown','pointerdown'].forEach(ev=>document.addEventListener(ev,crabWake,{passive:true}));
/* roaming: scuttle sideways along the composer when idle */
function roamNow(){const el=$('#crab');if(!el||ST.busy||CRAB.mood!=='idle'||CRAB.msg)return false;let x;do{x=.06+Math.random()*.88}while(Math.abs(x-CRAB.x)<.2);
const sec=Math.abs(x-CRAB.x)*4+.4;CRAB.mood='walk';paintCrab();crabGo(x,sec);clearTimeout(CRAB.rt);CRAB.rt=setTimeout(()=>{if(CRAB.mood==='walk'){CRAB.mood='idle';paintCrab()}},sec*1000+150);return true}
setInterval(()=>{if(!S||!S.settings.crab||!cx().roam||!cx().composer)return;if(Date.now()-CRAB.t>25000&&Math.random()<.45)roamNow()},7000);
setInterval(()=>{if(!S||!S.settings.crab||ST.busy||CRAB.mood!=='idle')return;const idle=Date.now()-CRAB.t;
if(idle>180000)setCrab('sleep',undefined,0,true);else if(idle>100000&&!CRAB.msg&&Math.random()<.5){setCrab('idle',pick(QUIPS[bucket()]),4500,true);CRAB.t-=60000}},20000);
/* eyes follow the mouse */
document.addEventListener('pointermove',e=>{if(!S||!S.settings.crab||!cx().look||ST.pm)return;ST.pm=requestAnimationFrame(()=>{ST.pm=0;const el=$('#crab');if(!el)return;const r=el.getBoundingClientRect(),dx=e.clientX-(r.left+r.width/2),dy=e.clientY-(r.top+r.height/2),ex=dx>90?1:dx<-90?-1:0,ey=dy>60?1:dy<-60?-1:0;
if(ex!==CRAB.ex||ey!==CRAB.ey){CRAB.ex=ex;CRAB.ey=ey;el.style.setProperty('--ex',ex);el.style.setProperty('--ey',ey)}})},{passive:true});
/* message avatars */
const avOn=()=>!!(S&&S.settings.crab&&cx().avatar);
const avFirst=prev=>!prev||(prev.role==='user'&&!prev.kind);
const avHtml=(m,prev)=>avOn()?`<div class="avt">${avFirst(prev)?miniCrab(m.streaming?moodFor(m.content,S.ui.tab==='code'):'idle',34):''}</div>`:'';
/* make `a` match `b` while keeping unchanged nodes alive, so running CSS animations are not restarted on every streamed token */
function morph(a,b){const an=[...a.childNodes],bn=[...b.childNodes];
for(let i=0;i<bn.length;i++){const x=an[i],y=bn[i];if(!x){a.appendChild(y.cloneNode(true));continue}
if(x.nodeType!==y.nodeType||x.nodeName!==y.nodeName){a.replaceChild(y.cloneNode(true),x);continue}
if(x.nodeType!==1){if(x.nodeValue!==y.nodeValue)x.nodeValue=y.nodeValue;continue}
for(const at of [...x.attributes])if(!y.hasAttribute(at.name))x.removeAttribute(at.name);for(const at of [...y.attributes])if(x.getAttribute(at.name)!==at.value)x.setAttribute(at.name,at.value);morph(x,y)}
while(a.childNodes.length>bn.length)a.removeChild(a.lastChild)}

/* ============ Discord Rich Presence (sync with the local server) ============ */
function rpcPayload(){const r=S.settings.rpc,code=S.ui.tab==='code',s=curSess(),e=curEffort(code?'code':'chat'),m=((ST.conn&&ST.conn.model)||'a local model').replace(/^koboldcpp\//,'');
const parts=[];if(r.showModel)parts.push(m);parts.push(e.name+' effort');if(code&&r.showProject&&s)parts.push(s.ws.split(/[\\/]/).pop());
return{enabled:!!r.on,client_id:r.cid||'',details:ST.busy?(code?'Building with an AI agent':'Chatting with a local AI'):(code?'Coding with a local AI':'Chatting with a local AI'),state:parts.join(' · '),large_image:r.img||'kobold',large_text:'KoboldCode'}}
const rpcStatus=r=>{const t={connected:'Connected'+(r.user?' as '+esc(r.user):''),waiting:'Waiting for Discord… '+esc(r.msg||''),error:esc(r.msg||'Error'),off:'Off'}[r.status]||esc(r.status);return `<div class="dstat ${esc(r.status)}"><i></i><span>${t}</span></div>`};
function rpcSync(force){if(!S||!S.settings.rpc)return;clearTimeout(ST.rt);ST.rt=setTimeout(()=>{const p=rpcPayload(),j=JSON.stringify(p);if(!force&&j===ST.rj)return;ST.rj=j;
api('/api/rpc',p).then(r=>{ST.rpc=r;const e=$('#rpcst');if(e)e.innerHTML=rpcStatus(r)}).catch(()=>{})},force?0:1500)}

/* ============ settings pages ============ */
const chk=(p,label,rr)=>`<label class="chk"><input type="checkbox" data-bind="${p}" ${getPath(S,p)?'checked':''} ${rr?'data-rr="1"':''}><span>${label}</span></label>`;
PAGES.look=()=>{const s=S.settings,sk=SKINS.find(x=>x.id===s.skin)||SKINS[0],groups=[...new Set(SKINS.map(x=>x.g))],bg=s.bg;
return `<h2>Appearance</h2><p class="mu">Pick a look. Everything below is saved with your settings.</p>
${groups.map(g=>`<div class="sg">${g}</div><div class="skg">${SKINS.filter(x=>x.g===g).map(x=>`<button class="sk ${x.id===sk.id?'on':''}" data-a="skin" data-id="${x.id}"><div class="sw3">${x.sw.map(c=>`<i style="background:${c}"></i>`).join('')}</div><b>${x.n}</b></button>`).join('')}</div>`).join('')}
${sk.id==='classic'?fld('Mode',sel('settings.theme',[['dark','Dark'],['light','Light'],['auto','Auto — light by day, dark by night']],1))+fld('Accent',`<div class="sw">${['#d97757','#e0a14a','#6aa86b','#4f9bd9','#9b7be0','#d9608f'].map(c=>`<button data-a="accent" data-c="${c}" class="${s.accent===c?'on':''}" style="background:${c}"></button>`).join('')}</div>`):chk('settings.accentAll','Use my accent colour on this theme',1)+(s.accentAll?fld('Accent',`<div class="sw">${['#d97757','#e0a14a','#6aa86b','#4f9bd9','#9b7be0','#d9608f'].map(c=>`<button data-a="accent" data-c="${c}" class="${s.accent===c?'on':''}" style="background:${c}"></button>`).join('')}</div>`):'')}
${sk.own?'':fld('Response font',sel('settings.font',[['serif','Serif (Claude-like)'],['sans','Sans-serif']],1))}
<div class="sg">Background</div><div class="wallg">${WALLS.map(([k,n])=>`<button class="wl ${(k==='none'&&bg.mode==='none')||(bg.mode==='preset'&&bg.preset===k)?'on':''}" data-a="wall" data-id="${k}" style="background:${k==='none'?'var(--card)':wallCss(k)};${k==='none'?'color:var(--tx);text-shadow:none':''}">${n}</button>`).join('')}<button class="wl ${bg.mode==='image'?'on':''}" data-a="upwall" style="background:var(--card2);color:var(--tx);text-shadow:none">${bg.mode==='image'?'Your image':'Upload image…'}</button></div>
${bg.mode==='image'?'<button class="btn dg" data-a="rmwall">Remove image</button>':''}
${bg.mode!=='none'?fld('Blur',rng('settings.bg.blur',0,30,1))+fld('Dim',rng('settings.bg.dim',0,80,1),'Darkens (or lightens, on light themes) the picture so text stays readable.')+fld('Panel opacity',rng('settings.bg.panel',30,100,1),'How much the sidebar and chat panels cover the picture.'):''}
<div class="sg">Effects</div>${fld('Animations',sel('settings.anim',[['full','Full'],['subtle','Subtle'],['off','Off']],1))}
<div class="f"><label>Effort colours</label><div class="efleg">${EFFORT.map(e=>`<span><i style="background:${e.col}"></i>${e.name}</span>`).join('')}</div><small>The composer, send button and glow follow the effort level.</small></div>
<div class="sg">Companion</div>${chk('settings.crab','Show Pinch, the pixel crab',1)}${s.crab?chk('settings.crabx.composer','Perch on the composer',1)+chk('settings.crabx.avatar','Avatar next to replies (acts out what the model is doing)',1)+chk('settings.crabx.roam','Wander along the composer when idle')+chk('settings.crabx.look','Eyes follow your mouse')+fld('Size',sel('settings.crabSize',[[48,'Small'],[64,'Medium'],[88,'Large']],1),'Pinch types, runs commands, cheers when tests pass, naps when idle, and dresses up with your effort level. Click it for a quip.'):''}`};
PAGES.discord=()=>{const r=S.settings.rpc;return `<h2>Discord</h2><p class="mu">Show what you're doing in KoboldCode on your Discord profile. Only generic text is shared — never your messages, code or file contents.</p>
${chk('settings.rpc.on','Enable Rich Presence')}
${fld('Application ID',bi('settings.rpc.cid'),'Required. Create a free app at discord.com/developers/applications → New Application, then copy its Application ID.')}
${fld('Image key',bi('settings.rpc.img'),'Optional. In your app: Rich Presence → Art Assets → upload <b>pinch.png</b> and name it exactly this key (default “kobold”).')}
${chk('settings.rpc.showModel','Show the model name')}${chk('settings.rpc.showProject','Show the project folder name (Code tab)')}
<div id="rpcst">${rpcStatus(ST.rpc||{status:'off'})}</div><button class="btn" data-a="rpctest">Reconnect / refresh</button>
<p class="mu" style="margin-top:14px">The Discord desktop app must be running on this computer. Updates are sent at most every 15 seconds.</p>`};
const FXACT={
skin(el){S.settings.skin=el.dataset.id;save();render()},
wall(el){const k=el.dataset.id;S.settings.bg.mode=k==='none'?'none':'preset';if(k!=='none')S.settings.bg.preset=k;save();render()},
upwall(){$('#bgf').click()},
async rmwall(){await api('/api/wallpaper',{clear:true});Object.assign(S.settings.bg,{mode:'none',v:0});save();render()},
crabpoke(){CRAB.t=Date.now();setCrab('happy',pick([...QUIPS[bucket()],'Pinch pinch!','I believe in your code!']),2600)},
async rpctest(){rpcSync(true);setTimeout(async()=>{try{ST.rpc=await api('/api/rpc');const e=$('#rpcst');if(e)e.innerHTML=rpcStatus(ST.rpc)}catch{}},1800)}};
setTimeout(()=>setCrab('idle',HELLO[bucket()],5200),1600);

/* ============ agent: tool-call protocol ============ */
const TN=['list_dir','read_file','write_file','edit_file','delete_path','search','run_command','todo','finish'],WRITE=['write_file','edit_file','delete_path'];
const OPEN=new RegExp('<('+TN.join('|')+')>');
const PK=['path','file_path','filename','content','old','old_str','new','new_str','command','cmd','pattern','items','summary','start','end','depth','timeout'],BIG=['content','old','new','old_str','new_str','items'];
function params(b,done){const o={};
for(const k of PK){const i=b.indexOf('<'+k+'>');if(i<0)continue;const s=i+k.length+2;let e=BIG.includes(k)?b.lastIndexOf('</'+k+'>'):b.indexOf('</'+k+'>',s);let partial=false;if(e<s){e=b.length;partial=true}
let v=b.slice(s,e);if(partial&&!done)v=v.replace(/<\/?[a-z_]*$/i,'');o[k]=v}
for(const[a,b2]of Object.entries({file_path:'path',filename:'path',old_str:'old',new_str:'new',cmd:'command'}))if(o[a]!=null&&o[b2]==null)o[b2]=o[a];
for(const k of['content','old','new'])if(o[k]!=null)o[k]=o[k].replace(/^\r?\n/,'').replace(/\r?\n[ \t]*$/,'');
for(const k of['path','command','pattern','start','end','depth','timeout'])if(o[k]!=null)o[k]=o[k].trim();
if(o.content!=null){const m=/^```[\w+-]*\n([\s\S]*?)\n?```\s*$/.exec(o.content);if(m)o.content=m[1]}
return o}
/* blank out <think>…</think> (same length, so indices still line up). Also handles templates that pre-open the think block. */
function thinkMask(raw){raw=raw||'';let out='',i=0;const c0=raw.indexOf('</think>'),o0=raw.indexOf('<think>');
if(c0>=0&&(o0<0||o0>c0)){out+=' '.repeat(c0+8);i=c0+8}
for(;;){const o=raw.indexOf('<think>',i);if(o<0){out+=raw.slice(i);break}out+=raw.slice(i,o);const c=raw.indexOf('</think>',o);if(c<0){out+=' '.repeat(raw.length-o);break}out+=' '.repeat(c+8-o);i=c+8}return out}
function parseTool(raw){raw=raw||'';const m=OPEN.exec(thinkMask(raw));if(!m)return null;const name=m[1],bs=m.index+m[0].length,close='</'+name+'>',ce=raw.indexOf(close,bs),complete=ce>=0;
return{name,args:params(raw.slice(bs,complete?ce:raw.length),complete),start:m.index,end:complete?ce+close.length:raw.length,complete}}
const parseTodos=t=>t.split('\n').map(l=>l.trim()).filter(Boolean).map(l=>{const m=/^[-*]?\s*\[([ xX~>]?)\]\s*(.*)$/.exec(l);return m?{d:/x/i.test(m[1]),t:m[2]}:{d:false,t:l.replace(/^[-*]\s*/,'')}});
const todoHtml=l=>`<ul class="todo">${l.map(x=>`<li class="${x.d?'d':''}">${x.d?ic('check'):'<i></i>'}<span>${esc(x.t)}</span></li>`).join('')}</ul>`;

/* ============ agent: cards ============ */
const codePrev=(c,pend)=>{const L=(c||'').split('\n');return pend?`<pre>${esc(L.slice(-12).join('\n'))}</pre>`:L.length<=14?`<pre>${esc(c)}</pre>`:`<details><summary>${L.length} lines</summary><pre>${esc(c)}</pre></details>`};
const diffHtml=(o,n)=>`<pre>${(o||'').split('\n').slice(0,40).map(l=>`<span class="dl r">- ${esc(l)}</span>`).join('')}${(n||'').split('\n').slice(0,40).map(l=>`<span class="dl a">+ ${esc(l)}</span>`).join('')}</pre>`;
function toolCard(t,pend){const a=t.args||{},n=t.name,M={list_dir:['folder','List',a.path||'.'],read_file:['file','Read',(a.path||'')+(a.start?` (${a.start}-${a.end||''})`:'')],write_file:['edit','Write',a.path],edit_file:['edit','Edit',a.path],delete_path:['trash','Delete',a.path],search:['search','Search',(a.pattern||'')+(a.path&&a.path!=='.'?'  in '+a.path:'')],run_command:['term','Run','$ '+(a.command||'')],todo:['list','Update plan',''],finish:['check','Done','']}[n];let body='';
if(n==='write_file'&&a.content!=null)body=codePrev(a.content,pend);else if(n==='edit_file'&&(a.old!=null||a.new!=null))body=diffHtml(a.old,a.new);
else if(n==='todo'&&a.items)body=todoHtml(parseTodos(a.items));else if(n==='finish'&&a.summary)body=`<div class="fin">${inl(a.summary)}</div>`;
return `<div class="tc"><div class="tch">${ic(M[0])}<b>${M[1]}</b><code>${esc(M[2]||'')}</code>${pend?'<span class="spin"></span>':''}</div>${body}</div>`}
function codeAsst(m){const pt=parseTool(m.content),text=pt?m.content.slice(0,pt.start):m.content;let h=text.trim()?renderMd(text.replace(/```\w*\s*$/,'')):'';
if(pt)h+=toolCard(pt,m.streaming&&!pt.complete);else if(m.streaming)h+=m.content?'<span class="cur"></span>':'<span class="dots"><i></i><i></i><i></i></span>';return h}
function toolRes(m){const out=m.content.replace(/^<tool_result[^>]*>\n?/,'').replace(/\n?<\/tool_result>$/,''),n=out?out.split('\n').length:0,open=m.tool.name==='run_command'&&out.length<1600;
return `<details class="tr ${m.ok?'ok':'er'}" ${open?'open':''}><summary>${ic(m.ok?'check':'x')} ${m.den?'Denied':m.ok?'Result':'Failed'}<span>${n} line${n===1?'':'s'}</span></summary><pre>${esc(out.slice(0,6000))}</pre></details>`}
function codeMsg(m,i,arr){if(m.kind==='tool')return `<div class="m" style="margin:0">${toolRes(m)}</div>`;if(m.kind==='nudge')return `<div class="note">${esc(m.note||'↪ Reminded the model to use its tools')}</div>`;if(m.kind==='sys')return `<div class="note">${esc(m.content)}</div>`;
if(m.role==='user')return userHtml(m);return `<div class="m a${avOn()?' av':''}${nw(m.id)}" data-mid="${m.id}" style="margin:14px 0">${avHtml(m,arr&&arr[i-1])}<div class="ab">${codeAsst(m)}</div></div>`}
function apprHtml(){const{s,t}=ST.appr,w=t.name==='run_command'?'run this command':t.name==='delete_path'?'delete this':'change this file';
return `<div class="appr"><div class="ph">${ic('shield')}&nbsp;Allow KoboldCode to ${w}?</div>${toolCard(t,false)}<div class="bs"><button class="btn pr" data-a="appr" data-v="allow">Allow</button><button class="btn" data-a="appr" data-v="always">Always allow this session</button><button class="btn dg" data-a="appr" data-v="deny">Deny</button></div></div>`}

/* ============ agent: prompts, tree, panel ============ */
function sysCode(s){const e=curEffort('code'),p=curPreset(),n=s.snap||{},plan=s.perm==='plan';
return `You are KoboldCode, an expert autonomous software engineer working inside the user's project folder. You build, edit, run and debug real projects by calling tools.

Workspace: ${n.cwd}
System: ${n.os}. Run Python with \`${n.py}\`. run_command uses the system shell inside the workspace.
Permission mode: ${PERMS[s.perm][0]}${plan?' - READ-ONLY. Only list_dir, read_file, search and todo work. Investigate, then answer with a clear numbered plan and stop.':''}

# Tools
Call exactly ONE tool per reply using this XML. After the closing tag STOP and wait for the <tool_result>. Never write a <tool_result> yourself. One sentence of explanation before the call is fine.

<list_dir><path>.</path></list_dir>
<read_file><path>src/app.py</path></read_file>    (optional <start>1</start><end>200</end>; line-number prefixes are not part of the file)
<write_file><path>src/app.py</path><content>
COMPLETE file contents here
</content></write_file>
<edit_file><path>src/app.py</path><old>exact existing lines</old><new>replacement lines</new></edit_file>
<delete_path><path>old.txt</path></delete_path>
<search><pattern>regex</pattern><path>.</path></search>
<run_command><command>${n.py} main.py</command></run_command>
<todo><items>
- [x] finished step
- [ ] next step
</items></todo>
<finish><summary>What you did and how to run it</summary></finish>

# Rules
- Paths are relative to the workspace. Explore with list_dir/read_file before changing existing code, and read a file before you edit it.
- write_file for new files and full rewrites (always the COMPLETE file, never "...rest of code"); edit_file for small precise changes.
- For work with 3+ steps keep a plan with todo and update it as steps complete.
- After writing code, run it or its tests with run_command and fix errors before finishing. Commands must exit on their own (no servers, no input()) and time out after 60s.
- When asked to build something, put the code in files with write_file, not in chat.
- When done, call finish. For plain questions just answer in text with no tool call.
- ${e.hint}${p.sys?'\n- '+p.sys:''}

# Project memory (KOBOLD.md)
${n.memory||'(none yet - the user can run /init)'}

# Workspace at session start
${n.tree||'(empty)'}`}
const ctxPct=s=>Math.round((est(sysCode(s))+s.msgs.reduce((a,m)=>a+est(m.content),0))/(+curPreset().ctx||8192)*100);
function ctxMeter(s){if(!s)return '';const p=ctxPct(s);return `<span id="ctxm" class="mu" style="font-size:12px;margin-right:6px;${p>=90?'color:var(--er)':p>=70?'color:#e5b94a':''}" title="Estimated context use">ctx ${Math.min(99,p)}%</span>`}
function ctxWarn(s){const p=ctxPct(s);if(p>=80&&!s.warned){s.warned=1;setCrab('ask',`Context ${p}% full — try /compact`,8000)}}
function treeHtml(s){const files=ST.tree[s.id]||[],ch=new Set(s.changed||[]),root={};
for(const f of files){let n=root;const ps=f.p.split('/');ps.forEach((x,i)=>{n=n[x]??={};if(i===ps.length-1)n.__f=f})}
const isF=n=>!!n.__f,walk=(n,path,d)=>Object.keys(n).filter(k=>k!=='__f').sort((a,b)=>(isF(n[a])-isF(n[b]))||a.localeCompare(b)).map(k=>{const c=n[k],p=path?path+'/'+k:k,pad=`style="padding-left:${4+d*12}px"`;
if(isF(c))return `<div class="tr1 ${ch.has(p)?'ch':''}" ${pad} data-a="openfile" data-p="${esc(p)}">${ic('file')}${esc(k)}</div>`;
const open=!ST.closed.has(s.id+p);return `<div class="tr1" ${pad} data-a="toggledir" data-p="${esc(s.id+p)}">${ic(open?'chev':'play')}${ic('folder')}${esc(k)}</div>`+(open?walk(c,p,d+1):'')}).join('');
return walk(root,'',0)||emptyHtml('Empty folder')}
const panelHtml=s=>`<aside class="cpanel" id="panel"><div class="ph">Plan</div>${s.todos?.length?todoHtml(s.todos):'<div class="empty" style="padding:0">The agent keeps its plan here.</div>'}<div class="ph">Files<button class="ib" data-a="reftree" title="Refresh">${ic('refresh')}</button></div><div class="tree">${treeHtml(s)}</div></aside>`;
async function refreshTree(s){try{const r=await api('/api/ws/tree?path='+enc(s.ws));ST.tree[s.id]=r.files||[]}catch{}const p=$('#panel');if(p&&curSess()===s)p.outerHTML=panelHtml(s)}
async function openFile(s,rel){const r=await api(`/api/ws/file?path=${enc(s.ws)}&rel=${enc(rel)}`);
modal(`<h3>${esc(rel)}</h3><div class="viewer">${r.ok?`<pre>${hl(r.text,rel.split('.').pop())}</pre>`:`<div class="er">${esc(r.err||r.error)}</div>`}</div><div class="mb"><a class="btn" href="/api/ws/file?path=${enc(s.ws)}&amp;rel=${enc(rel)}&amp;dl=1&amp;t=${TOKEN}" download>${ic('dl')} Download</a><button class="btn pr" data-a="mcancel">Close</button></div>`,1)}
const exportUrl=s=>`/api/ws/export?path=${enc(s.ws)}&t=${TOKEN}`;

/* ============ agent: views ============ */
const IDEAS=[['Snake game','Build a polished Snake game as a single index.html with canvas graphics, score, speed levels and keyboard controls.'],['Python CLI','Create a Python command-line todo manager with argparse, JSON storage, colored output and unit tests.'],['FastAPI service','Build a small FastAPI notes API with SQLite storage, pydantic models, tests and a README.'],['Landing page','Design a responsive landing page for a coffee roastery in plain HTML and CSS with a distinctive look.']];
function viewCodeHome(){const b=bucket(),nm=S.settings.name?', '+S.settings.name:'';ST.cg??=CG[b].replace('{n}',nm);
return topbar('<span class="sp"></span>')+`<div class="home" style="--glow:${GLOW[b]}"><div class="hello">${LOGO(40)}<h1>${esc(ST.cg)}</h1></div><div class="sub" id="clock">${dayLine()}</div>
<div class="wsrow">${ic('folder')}<input id="ws" placeholder="Workspace folder — leave empty to create a new project in ${esc(META.ws)}" value="${esc(ST.wsDraft||'')}"><button class="btn" data-a="browse">Browse…</button></div>
${composer('code','What should we build? (type / for commands)')}<div class="chips">${S.recentWs.map(w=>`<button class="chip" data-a="setws" data-w="${esc(w)}">${ic('folder')}${esc(w.split(/[\\/]/).pop())}</button>`).join('')}${IDEAS.map(c=>`<button class="chip" data-a="chip" data-t="${esc(c[1])}">${esc(c[0])}</button>`).join('')}</div></div>`}
function viewCode(s){const last=s.msgs[s.msgs.length-1],plan=s.perm==='plan'&&!ST.busy&&last&&last.role==='assistant'&&last.content.trim()&&!parseTool(last.content);
return topbar(`<div class="ttl"><b>${esc(s.title)}</b><span class="ws" data-a="openws" title="Open folder">${esc(s.ws)}</span></div><span class="sp"></span>${ctxMeter(s)}<button class="ib" data-a="undo" title="Undo last file change">${ic('undo')}</button><a class="ib" href="${exportUrl(s)}" download title="Export workspace as .zip">${ic('dl')}</a><button class="ib ${ST.panel?'on':''}" data-a="panel" title="Plan & files">${ic('panel')}</button>`)+
`<div class="cwrap"><div class="cmain"><div class="msgs" id="msgs"><div class="col">${s.msgs.map(codeMsg).join('')}${ST.appr&&ST.appr.s===s?apprHtml():''}${plan?'<div class="bs" style="margin:10px 0"><button class="btn pr" data-a="execplan">'+ic('play')+' Approve plan & build</button></div>':''}</div></div>
<div class="dock"><div class="col">${composer('code','Describe what to build or change… (type / for commands)')}<div class="fine">Esc interrupts · Shift+Enter for a new line</div></div></div></div>${ST.panel?panelHtml(s):''}</div>`}

/* ============ agent: execution ============ */
const sysNote=(s,t)=>s.msgs.push({id:uid(),role:'user',kind:'sys',content:t});
const TOOL_REMINDER='\n\n[Reminder: when asked to build or change something, create and edit files with write_file / edit_file and test with run_command. Do not paste whole files into chat. One tool call per reply.]';

/* ---- when is a pasted code block a real problem? Only if it is a file-sized dump that was NOT saved and the model has not been using its tools ---- */
const NOFILE=/^(|bash|sh|shell|zsh|console|terminal|text|txt|output|log|powershell|ps1|cmd|bat|diff|plaintext)$/i;
const isCont=m=>m.role==='user'&&!m.kind&&(m.text||m.content||'').trim().length<25;
function turnStart(s){let i=s.msgs.length-1;for(;i>=0;i--){const m=s.msgs[i];if(m.role==='user'&&!m.kind&&!isCont(m))break}return Math.max(i,0)}
const lastPrompt=s=>{const m=s.msgs[turnStart(s)];return m&&m.role==='user'?(m.text||m.content||''):''};
const deniedThisTurn=s=>s.msgs.slice(turnStart(s)).some(m=>m.kind==='tool'&&m.den);
const workedThisTurn=s=>s.msgs.slice(turnStart(s)).some(m=>m.kind==='tool'&&m.ok&&(WRITE.includes(m.tool.name)||m.tool.name==='run_command'));
const questionish=t=>/\b(explain|why|what is|what does|what are|how does|how do i|difference|compare|example of|walk me through|describe|show me how)\b/i.test(t)&&!/\b(create|build|make|write|implement|generate|app|script|program|game|fix|add|refactor|save)\b/i.test(t);
function savedLines(s){const set=new Set();for(const m of s.msgs)if(m.kind==='tool'&&m.ok&&m.tool){const a=m.tool.args||{},t=m.tool.name==='write_file'?a.content:m.tool.name==='edit_file'?a.new:'';(t||'').split('\n').forEach(l=>{l=l.trim();if(l.length>2)set.add(l)})}return set}
function looksSaved(saved,body){const L=body.split('\n').map(l=>l.trim()).filter(l=>l.length>2);if(L.length<3||!saved)return false;return L.filter(l=>saved.has(l)).length/L.length>=.6}
/* returns true when `c` contains a big, unsaved, file-like code block outside any <think> section */
function unsavedBig(c,min,saved){const m=/```([\w+#.-]*)[^\n]*\n([\s\S]*?)(```|$)/.exec(thinkMask(c));return !!m&&m[2].length>=min&&!NOFILE.test(m[1])&&!looksSaved(saved,m[2])}
const toHist=(m,perm)=>m.filter(x=>x.kind!=='sys').map(x=>({role:x.role,content:x.role==='user'&&!x.kind&&perm!=='plan'?x.content+TOOL_REMINDER:x.content}));
const needsApproval=(s,t)=>t.name==='run_command'?s.perm!=='yolo':WRITE.includes(t.name)&&s.perm==='ask';
const askApproval=(s,t)=>new Promise(res=>{ST.appr={s,t,res};setCrab('ask','May I?');render()});
async function execTool(s,t){const n=t.name,a=t.args||{};
if(n==='todo'){s.todos=parseTodos(a.items||'');return{ok:true,out:'Plan updated. Continue with the next unfinished step.'}}
if(s.perm==='plan'&&(WRITE.includes(n)||n==='run_command'))return{ok:false,out:'PLAN MODE is read-only: you cannot modify files or run commands. Finish investigating, then give your numbered plan in plain text.'};
if(needsApproval(s,t)){const d=await askApproval(s,t);if(d==='deny')return{ok:false,den:1,out:'The user denied this action. Do not retry it; ask what they would like instead.'};if(d==='always')s.perm=n==='run_command'?'yolo':'auto'}
if(ST.stopReq)return{ok:false,out:'Interrupted by the user.',stop:1};
const r=await api('/api/tool',{ws:s.ws,name:n,args:a});
if(WRITE.includes(n)&&r.ok){const p=(a.path||'').replace(/^\.\//,'');s.changed=[...new Set([...(s.changed||[]),p])]}
if(WRITE.includes(n)||n==='run_command')await refreshTree(s);return r}
async function codeLoop(s){const e=curEffort('code');ST.busy=true;ST.stopReq=false;let nudged=false,cuts=0;setCrab('think');
const nudge=(content,note)=>s.msgs.push({id:uid(),role:'user',kind:'nudge',content,note});
try{for(let step=0;step<e.steps;step++){const a={id:uid(),role:'assistant',content:'',streaming:true};s.msgs.push(a);render();let hit=null,big=false;
/* only police pasted code when the model has not been doing real work this turn, and the question is not just "explain …" */
const police=s.perm!=='plan'&&!nudged&&!workedThisTurn(s)&&!deniedThisTurn(s)&&!questionish(lastPrompt(s)),saved=police?savedLines(s):null;
await doStream(mkReq(sysCode(s),toHist(s.msgs.slice(0,-1),s.perm),{code:true}),t=>{a.content+=t;const pt=parseTool(a.content);crabForTool(pt,a.content);if(pt&&pt.complete){hit=pt;return false}
if(police&&!pt&&unsavedBig(a.content,1200,saved)){big=true;return false}paint(a)});
a.streaming=false;if(hit)a.content=a.content.slice(0,hit.end);
if(ST.stopReq){if(!a.content.trim())s.msgs.pop();break}
const vis=thinkMask(a.content);
if(big){nudged=true;a.content=a.content.slice(0,vis.indexOf('```')).trim();if(!a.content)s.msgs.pop();
nudge('Do not paste code into the chat. Call write_file with the complete file now (one tool call), then run it with run_command to verify.');continue}
if(!hit){const pt=parseTool(a.content);
if(pt&&!pt.complete){/* the reply hit the token limit in the middle of a tool call: nothing ran */
if(cuts++<2){a.content=a.content.slice(0,pt.start).trim();if(!a.content)s.msgs.pop();
nudge('Your last reply hit the token limit in the middle of a tool call, so nothing was executed. Retry in smaller steps: create the file as a short skeleton first, then add the rest with edit_file in several parts. One tool call per reply.','↪ Reply was cut off mid-tool-call — asking the model to retry in smaller pieces');continue}
sysNote(s,'The model keeps running out of tokens in the middle of a file. Raise the effort level (longer replies) or ask for smaller files.');break}
if(!vis.trim()){if(!a.content.trim())s.msgs.pop();sysNote(s,a.content.trim()?'The reply ended while the model was still thinking — the token budget for this effort level ran out. Try a higher effort, or type "continue".':'The model returned an empty reply. Try again or change the preset.');break}
if(police&&unsavedBig(a.content,600,saved)){nudged=true;nudge('You wrote code in the message instead of using a tool. Use write_file to create or change files and run_command to run them, then continue.');continue}
break}
if(hit.name==='finish')break;
const r=await execTool(s,hit);r.out=String(r.out??r.error??'tool failed');
s.msgs.push({id:uid(),role:'user',kind:'tool',tool:{name:hit.name,args:hit.args},ok:!!r.ok,den:r.den,content:`<tool_result name="${hit.name}">\n${r.out}\n</tool_result>`});
crabAfter(hit,r);ctxWarn(s);if(r.stop)break;if(step===e.steps-1)sysNote(s,`Step limit (${e.steps}) reached at ${e.name} effort. Say "continue" to keep going.`)}}
catch(err){sysNote(s,'Error: '+err.message+' — is KoboldCpp running? (Customize → Connection)')}
{const last=s.msgs[s.msgs.length-1];if(ST.stopReq)setCrab('idle','');else if(last&&last.kind==='sys'&&/^Error|empty reply|still thinking|running out of tokens/.test(last.content))setCrab('sad','Something went wrong',5200);else setCrab('happy',pick(['All done!','Shipped it!','Nailed it!']),5200)}
ST.busy=false;ST.stopReq=false;s.updated=Date.now();save();await refreshTree(s);render()}
const slug=t=>t.toLowerCase().replace(/[^a-z0-9 ]+/g,'').trim().split(/\s+/).slice(0,4).join('-');
async function codeSend(text,files){if(text.startsWith('/')){const r=await slash(text);if(r===null)return;text=r}
let s=curSess();
if(!s){const ws=(ST.wsDraft||'').trim()||META.ws+META.sep+(slug(text)||'project-'+uid().slice(0,4)),snap=await api('/api/ws/context?path='+enc(ws));if(snap.error){toast(snap.error);return}
s={id:uid(),title:titleFrom(text),ws:snap.cwd,snap,msgs:[],todos:[],perm:S.settings.perm,changed:[],pinned:false,created:Date.now(),updated:Date.now()};
S.sessions.push(s);S.ui.sess=s.id;S.ui.view='sess';S.recentWs=[snap.cwd,...S.recentWs.filter(x=>x!==snap.cwd)].slice(0,5);ST.wsDraft='';ST.cg=null}
const body=files.map(f=>`<attachment name="${f.name}">\n${f.text}\n</attachment>`).join('\n')+(files.length?'\n\n':'')+text;
s.msgs.push({id:uid(),role:'user',content:body,text,files:files.map(f=>f.name)});save();refreshTree(s);codeLoop(s)}

/* ============ slash commands ============ */
const MACRO={init:'Explore this workspace (list_dir, then read the key files). Then create KOBOLD.md in the workspace root: concise project memory covering purpose, structure, how to run and test, and coding conventions. Keep it under 60 lines.',
review:'Review the code in this workspace for bugs, edge cases, security problems and style. Read the important files, report concrete issues with file and line, and fix the clear ones.',
test:'Find how tests are run in this project (or write a minimal test suite if none exists), run them with run_command, and fix any failures.',
commit:'Run git status and git diff (run git init first if this is not a repository), then create a git commit with a clear, concise message.',
readme:'Read the project and write a clear README.md: what it is, how to install, run and test it, with usage examples.',
explain:'Explain how this codebase works: structure, main flow and key files. Do not change anything.'};
const CMDS={help:'Show all commands',init:'Create KOBOLD.md project memory',review:'Review code for bugs and fix them',test:'Run tests and fix failures',commit:'Create a git commit',readme:'Write a README',explain:'Explain the codebase',plan:'Toggle plan mode (read-only)',mode:'Set mode: ask | auto | plan | yolo',effort:'Set effort: low | medium | high | extra | ultracode',undo:'Undo last file change',export:'Download workspace as .zip',compact:'Shrink old tool output to save context',clear:'Clear this session',open:'Open workspace folder'};
async function slash(text){const[c,...r]=text.slice(1).split(/\s+/),arg=r.join(' '),s=curSess();
if(MACRO[c])return MACRO[c]+(arg?'\n\nExtra instructions: '+arg:'');const mac=S.macros.find(m=>m.n===c);if(mac)return mac.p.replace(/\$ARGS/g,arg);
const need=()=>{if(!s)toast('Start a session first');return !s};
switch(c){
case 'help':modal(`<h3>Slash commands</h3>${Object.entries(CMDS).concat(S.macros.map(m=>[m.n,'Custom'])).map(([k,v])=>`<div class="row"><code style="min-width:90px">/${esc(k)}</code><span class="mu">${esc(v)}</span></div>`).join('')}<div class="mb"><button class="btn pr" data-a="mcancel">Close</button></div>`);break;
case 'plan':case 'mode':{const m=c==='plan'?((s?s.perm:S.settings.perm)==='plan'?'ask':'plan'):arg;if(!PERMS[m]){toast('Modes: ask, auto, plan, yolo');break}if(s)s.perm=m;else S.settings.perm=m;save();render();toast('Mode: '+PERMS[m][0]);break}
case 'effort':{if(!EFFORT.find(e=>e.id===arg)){toast('Levels: '+EFFORT.map(e=>e.id).join(', '));break}S.settings.ceffort=arg;save();render();toast('Effort: '+arg);break}
case 'undo':if(!need())ACT.undo();break;
case 'export':if(!need()){const a=document.createElement('a');a.href=exportUrl(s);a.download='';a.click()}break;
case 'open':if(!need())api('/api/open',{path:s.ws});break;
case 'clear':if(!need()){s.msgs=[];s.todos=[];s.warned=0;save();render();toast('Session cleared')}break;
case 'compact':if(!need()){s.warned=0;s.msgs.forEach(m=>{if(m.kind==='tool'&&m.content.length>400)m.content=m.content.slice(0,300)+'\n…[compacted]\n</tool_result>'});save();render();toast('Old tool output compacted')}break;
default:toast(`Unknown command /${c} — type /help`)}
return null}

/* ============ render ============ */
const viewKey=()=>S.ui.tab+S.ui.view+(S.ui.chat||'')+(S.ui.sess||'')+(S.ui.proj||'');
function mainView(){const u=S.ui;if(u.view==='custom')return viewCustom();
if(u.tab==='chat'){if(u.view==='projects')return viewProjects();if(u.view==='project'){const p=S.projects.find(x=>x.id===u.proj);if(p)return viewProject(p)}const c=curChat();return c?viewChat(c):viewHome()}
const s=curSess();return s?viewCode(s):viewCodeHome()}
function render(){applyTheme();const vk=viewKey(),changed=ST.vk!==vk;
if(changed){const c=curChat()||curSess();if(c)c.msgs.forEach(m=>ST.seen.add(m.id))}
const m=$('#msgs'),keep=m?{top:m.scrollTop,near:m.scrollHeight-m.scrollTop-m.clientHeight<100}:null;
$('#app').innerHTML=sidebar()+`<main class="main${changed?' enter':''}">${mainView()}</main>`;
const inp=$('#inp');if(inp){inp.value=ST.draft||'';autosize(inp);if($('#overlay').className!=='on')inp.focus()}
const m2=$('#msgs');if(m2){if(keep&&ST.vk===vk&&!keep.near)m2.scrollTop=keep.top;else m2.scrollTop=m2.scrollHeight}ST.vk=vk;
$$('[data-mid]').forEach(e=>ST.seen.add(e.dataset.mid));
const s=curSess();if(s&&!ST.tree[s.id]){ST.tree[s.id]=[];refreshTree(s)}rpcSync()}
async function checkConn(){try{ST.conn=await api(`/api/test?url=${enc(S.settings.url)}&key=${enc(S.settings.key)}`)}catch{ST.conn={ok:false,err:'Local server unreachable'}}
const c=$('#conn');if(c)c.outerHTML=connHtml();const r=$('#testres'),x=ST.conn;
if(r)r.innerHTML=x.ok?`<div class="ok">${ic('check')} Connected — ${esc((x.model||'').replace(/^koboldcpp\//,''))}${x.ctx?` · context ${x.ctx}`:''}${x.version?` · KoboldCpp ${x.version}`:''}</div>${x.ctx?`<button class="btn" style="margin-top:8px" data-a="usectx" data-v="${x.ctx}">Set active preset context to ${x.ctx}</button>`:''}`:`<div class="er">${ic('x')} ${esc(x.err||'Failed')}</div>`}
function confirmBox(msg,fn){modal(`<h3>${esc(msg)}</h3><div class="mb"><button class="btn" data-a="mcancel">Cancel</button><button class="btn pr" data-a="mconfirm">Confirm</button></div>`);ST.mfn=fn}
const readFiles=async fs=>{const out=[];for(const f of fs){if(f.size>400000){toast(f.name+' is too large (max 400 KB)');continue}const text=await f.text();if(text.includes('\u0000')){toast(f.name+' looks binary');continue}out.push({name:f.name,text})}return out};
const EXT={python:'py',py:'py',javascript:'js',js:'js',typescript:'ts',ts:'ts',tsx:'tsx',jsx:'jsx',html:'html',css:'css',json:'json',bash:'sh',sh:'sh',rust:'rs',go:'go',java:'java',c:'c',cpp:'cpp',csharp:'cs',cs:'cs',sql:'sql',yaml:'yml',md:'md',markdown:'md'};

/* ============ actions ============ */
const ACT={
tab(el){const t=el.dataset.v,u=S.ui;u.tab=t;u.view=t==='chat'?(S.chats.find(c=>c.id===u.chat)?'chat':'home'):(S.sessions.find(c=>c.id===u.sess)?'sess':'home');ST.files=[];ST.draft='';save();render()},
new(){const u=S.ui;u.view='home';if(u.tab==='chat'){u.chat=null;u.proj=null}else u.sess=null;ST.greet=ST.cg=null;ST.files=[];ST.draft='';save();render();document.body.classList.remove('sb')},
search(){openSearch()},
projects(){S.ui.tab='chat';S.ui.view='projects';save();render()},
custom(){S.ui.view='custom';save();render()},
goconn(){S.ui.view='custom';S.ui.cpage='conn';save();render();checkConn()},
togside(){document.body.classList.toggle('sb')},
openchat(el){closeModal();const u=S.ui;u.tab='chat';u.view='chat';u.chat=el.dataset.id;ST.draft='';save();render();document.body.classList.remove('sb')},
opensess(el){closeModal();const u=S.ui;u.tab='code';u.view='sess';u.sess=el.dataset.id;ST.draft='';save();render();document.body.classList.remove('sb')},
openproject(el){closeModal();const u=S.ui;u.tab='chat';u.view='project';u.proj=el.dataset.id;save();render()},
itemmenu(el){const k=el.dataset.k,id=el.dataset.id,arr=k==='code'?S.sessions:S.chats,it=arr.find(x=>x.id===id);if(!it)return;
const items=[{label:it.pinned?'Unpin':'Pin to top',icon:'pin',fn:()=>{it.pinned=!it.pinned;save();render()}},{label:'Rename',icon:'edit',fn:async()=>{const v=await askText('Rename',it.title);if(v&&v.trim()){it.title=v.trim();save();render()}}}];
if(k==='chat'&&S.projects.length)items.push({label:'Move to project',icon:'folder',fn:()=>menu(el,[{label:'No project',on:!it.projectId,fn:()=>{it.projectId=null;save();render()}},...S.projects.map(p=>({label:p.name,on:it.projectId===p.id,fn:()=>{it.projectId=p.id;save();render()}}))])});
items.push({sep:1},{label:'Delete',icon:'trash',danger:1,fn:()=>{arr.splice(arr.indexOf(it),1);if(S.ui.chat===id||S.ui.sess===id){S.ui.view='home';S.ui.chat=S.ui.sess=null}save();render()}});menu(el,items)},
chip(el){const t=$('#inp');if(!t)return;t.value=ST.draft=el.dataset.t;autosize(t);t.focus()},
async send(){if(ST.busy)return toast('Stop the current response first');const t=$('#inp');if(!t)return;const text=t.value.trim();if(!text&&!ST.files.length)return;
const files=ST.files;ST.files=[];ST.draft='';t.value='';closeMenu();return S.ui.tab==='code'?codeSend(text,files):chatSend(text,files)},
stop(){ST.stopReq=true;abortGen();if(ST.appr){const r=ST.appr;ST.appr=null;r.res('deny')}},
attach(){$('#fi').click()},
rmfile(el){ST.files.splice(+el.dataset.i,1);$('#files').innerHTML=filesHtml()},
presetmenu(el){menu(el,[...S.presets.map(p=>({label:p.name,hint:p.mode==='native'?'native':'',on:p.id===S.settings.preset,fn:()=>{S.settings.preset=p.id;save();render()}})),{sep:1},{label:'Edit presets…',icon:'sliders',fn:()=>{S.ui.view='custom';S.ui.cpage='presets';save();render()}}],true)},
effortmenu(el){const k=S.ui.tab==='code'?'ceffort':'effort';menu(el,EFFORT.map(e=>({label:e.name,hint:e.desc,icon:'bolt',col:e.col,on:S.settings[k]===e.id,fn:()=>{S.settings[k]=e.id;save();render()}})),true)},
permmenu(el){const s=curSess();menu(el,Object.entries(PERMS).map(([k,v])=>({label:v[0],hint:v[1],on:(s?s.perm:S.settings.perm)===k,fn:()=>{if(s)s.perm=k;else S.settings.perm=k;save();render()}})),true)},
copymsg(el){const c=curChat()||curSess(),m=c&&c.msgs.find(x=>x.id===el.dataset.id);if(m){navigator.clipboard.writeText(fixThink(m.content).replace(/<think>[\s\S]*?(<\/think>|$)/g,'').trim());toast('Copied')}},
regen(){const c=curChat();if(!c||ST.busy)return;c.msgs.pop();runChat(c)},
copycode(el){navigator.clipboard.writeText($('code',el.closest('.cb')).textContent);toast('Copied')},
savecode(el){const l=(el.dataset.l||'').toLowerCase();download('snippet.'+(EXT[l]||l||'txt'),$('code',el.closest('.cb')).textContent)},
async savews(el){const s=curSess();if(!s)return;const l=(el.dataset.l||'').toLowerCase(),text=$('code',el.closest('.cb')).textContent,n=await askText('Save to workspace as',l==='html'?'index.html':'main.'+(EXT[l]||l||'txt'));
if(!n||!n.trim())return;const r=await api('/api/tool',{ws:s.ws,name:'write_file',args:{path:n.trim(),content:text}});toast(r.out||r.error||'failed');if(r.ok){s.changed=[...new Set([...(s.changed||[]),n.trim()])];save();refreshTree(s)}},
/* projects */
newproj(){projModal()},editproj(el){projModal(S.projects.find(p=>p.id===el.dataset.id))},
saveproj(el){const name=$('#pn').value.trim();if(!name)return toast('Give the project a name');let p=S.projects.find(x=>x.id===el.dataset.id);
if(!p){p={id:uid(),files:[],created:Date.now()};S.projects.push(p);S.ui.view='project';S.ui.proj=p.id}p.name=name;p.desc=$('#pd').value.trim();p.instr=$('#pi').value;closeModal();save();render()},
delproj(el){confirmBox('Delete this project? Its chats are kept.',()=>{S.projects=S.projects.filter(p=>p.id!==el.dataset.id);S.chats.forEach(c=>{if(c.projectId===el.dataset.id)c.projectId=null});S.ui.view='projects';save();render()})},
addpfile(el){ST.pfid=el.dataset.id;$('#pfi').click()},
rmpfile(el){const p=S.projects.find(x=>x.id===el.dataset.id);p.files.splice(+el.dataset.i,1);save();render()},
/* customize */
cpage(el){S.ui.cpage=el.dataset.v;save();render();if(el.dataset.v==='conn')checkConn()},
accent(el){S.settings.accent=el.dataset.c;save();render()},
testconn(){$('#testres').textContent='Testing…';checkConn()},
usectx(el){curPreset().ctx=+el.dataset.v;save();render();toast('Context set to '+el.dataset.v)},
usepreset(el){S.settings.preset=el.dataset.id;save();render()},
duppreset(){const p=curPreset(),n={...p,id:uid(),name:p.name+' copy'};S.presets.push(n);S.settings.preset=n.id;save();render()},
delpreset(){if(S.presets.length<2)return toast('Keep at least one preset');S.presets=S.presets.filter(p=>p!==curPreset());S.settings.preset=S.presets[0].id;save();render()},
resetpresets(){confirmBox('Reset all presets to defaults?',()=>{S.presets=PRESETS();S.settings.preset='balanced';save();render()})},
addmacro(){S.macros.push({n:'',p:''});save();render()},delmacro(el){S.macros.splice(+el.dataset.i,1);save();render()},
exportall(){download('koboldcode-backup.json',JSON.stringify(S,null,1),'application/json')},importall(){$('#imp').click()},
clearchats(){confirmBox('Delete ALL chats and code sessions?',()=>{S.chats=[];S.sessions=[];S.ui.view='home';S.ui.chat=S.ui.sess=null;save();render()})},
/* modal */
mok(){const v=$('#mt').value;closeModal();const f=ST.mres;ST.mres=null;f&&f(v)},
mcancel(){closeModal();const f=ST.mres;ST.mres=null;f&&f(null)},
mconfirm(){const f=ST.mfn;closeModal();f&&f()},
closebg(el,e){if(e.target===el)ACT.mcancel()},
/* code */
async browse(){const r=await api('/api/pickdir',{});if(r.path){ST.wsDraft=r.path;const w=$('#ws');if(w)w.value=r.path}},
setws(el){ST.wsDraft=el.dataset.w;const w=$('#ws');if(w)w.value=el.dataset.w},
openws(){api('/api/open',{path:curSess().ws})},
async undo(){const s=curSess();if(!s)return;const r=await api('/api/undo',{ws:s.ws});toast(r.out);await refreshTree(s)},
panel(){ST.panel=!ST.panel;render()},reftree(){refreshTree(curSess())},
openfile(el){openFile(curSess(),el.dataset.p)},
toggledir(el){const k=el.dataset.p;ST.closed.has(k)?ST.closed.delete(k):ST.closed.add(k);$('#panel').outerHTML=panelHtml(curSess())},
appr(el){const r=ST.appr;ST.appr=null;r.res(el.dataset.v);render()},
execplan(){const s=curSess();s.perm='auto';save();codeSend('The plan is approved. Implement it now using your tools.',[])}};
Object.assign(ACT,FXACT);

/* ============ events ============ */
document.addEventListener('click',e=>{const mi=e.target.closest('#menu .mi');
if(mi){const it=$('#menu')._items[mi.dataset.i];closeMenu();if(it&&it.fn)it.fn();return}
if(!e.target.closest('#menu')){closeMenu();ST.slashOpen=false}
const t=e.target.closest('[data-a]');if(t&&ACT[t.dataset.a])ACT[t.dataset.a](t,e)});
function slashHint(t){const v=t.value;if(S.ui.tab!=='code'||!/^\/\w*$/.test(v)){if(ST.slashOpen){closeMenu();ST.slashOpen=false}return}
const all=[...Object.entries(CMDS),...S.macros.filter(m=>m.n).map(m=>[m.n,'Custom command'])].filter(([k])=>k.startsWith(v.slice(1)));if(!all.length){closeMenu();return}
ST.slashOpen=true;menu($('.comp'),all.map(([k,d])=>({label:'/'+k,hint:d,fn:()=>{t.value=ST.draft='/'+k+' ';t.focus();autosize(t);ST.slashOpen=false}})),true)}
document.addEventListener('input',e=>{const t=e.target;
if(t.id==='inp'){ST.draft=t.value;autosize(t);slashHint(t);return}if(t.id==='ws'){ST.wsDraft=t.value;return}
const b=t.dataset&&t.dataset.bind;if(!b)return;let v=t.type==='checkbox'?t.checked:t.type==='range'||t.type==='number'?parseFloat(t.value):t.value;if(typeof v==='number'&&isNaN(v))v=0;
setPath(S,b,v);const o=t.nextElementSibling;if(o&&o.tagName==='OUTPUT')o.textContent=v;save();if(b.startsWith('settings.'))applyTheme();
if(b==='settings.url'||b==='settings.key'){clearTimeout(ST.ct);ST.ct=setTimeout(checkConn,800)}if(b.startsWith('settings.rpc'))rpcSync(true);if((t.tagName==='SELECT'||t.type==='checkbox')&&t.dataset.rr)render()});
document.addEventListener('change',async e=>{const t=e.target;
if(t.id==='fi'){ST.files.push(...await readFiles(t.files));t.value='';const el=$('#files');if(el)el.innerHTML=filesHtml()}
else if(t.id==='pfi'){const p=S.projects.find(x=>x.id===ST.pfid);if(p){(p.files??=[]).push(...await readFiles(t.files));save();render()}t.value=''}
else if(t.id==='imp'){try{const d=JSON.parse(await t.files[0].text()),D=defState();S={...D,...d,settings:{...D.settings,...d.settings},ui:D.ui};save();render();toast('Imported')}catch{toast('That file is not a valid backup')}t.value=''}});
document.addEventListener('keydown',e=>{
if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();return}
if(e.key==='Escape'){if($('#overlay').className==='on')return ACT.mcancel();if($('#menu').className==='on')return closeMenu();if(ST.busy)return ACT.stop()}
if(e.target.id==='inp'){if(e.key==='Tab'&&ST.slashOpen){e.preventDefault();const it=$('#menu')._items;if(it&&it[0]){closeMenu();it[0].fn()}return}
if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();ACT.send()}}});

/* ============ boot ============ */
async function boot(){const r=await api('/api/state');META={...META,...(r.meta||{})};const d=r.state||{},D=defState();
S={...D,...d,settings:{...D.settings,...(d.settings||{})},ui:{...D.ui,...(d.ui||{})}};S.settings.bg={...D.settings.bg,...S.settings.bg};S.settings.rpc={...D.settings.rpc,...S.settings.rpc};S.settings.crabx={...D.settings.crabx,...S.settings.crabx};if(!S.presets||!S.presets.length)S.presets=D.presets;
ST.closed=new Set();ST.draft='';[...S.chats,...S.sessions].forEach(c=>c.msgs.forEach(m=>{m.streaming=false}));
const u=S.ui;if(u.view==='chat'&&!S.chats.find(c=>c.id===u.chat))u.view='home';if(u.view==='sess'&&!S.sessions.find(c=>c.id===u.sess))u.view='home';if(u.view==='project'&&!S.projects.find(p=>p.id===u.proj))u.view='projects';
render();checkConn();setInterval(()=>{const c=$('#clock');if(c)c.textContent=dayLine()},30000);setInterval(applyTheme,300000)}
boot().catch(e=>{document.body.innerHTML='<pre style="padding:24px;color:#e5726a;white-space:pre-wrap">KoboldCode failed to start: '+e+'</pre>'});
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def send_bytes(self, data, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def j(self, obj, code=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", code)

    def host_ok(self):
        if CFG["host"] not in ("127.0.0.1", "localhost"):
            return True
        return self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]") in ("127.0.0.1", "localhost", "::1")

    def authed(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        tok = self.headers.get("X-KC") or (q.get("t") or [""])[0]
        return self.host_ok() and secrets.compare_digest(tok, TOKEN)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/":
            if not self.host_ok():
                return self.j({"error": "bad host"}, 403)
            html = UI.replace("__TOKEN__", TOKEN).replace("__KOBOLD__", CFG["kobold"])
            return self.send_bytes(html.encode(), "text/html; charset=utf-8", extra={"Cache-Control": "no-store"})
        if not self.authed():
            return self.j({"error": "unauthorized"}, 401)
        try:
            if u.path == "/api/state":
                st = json.loads(STATE.read_text("utf-8")) if STATE.exists() else None
                return self.j({"state": st, "meta": {"ws": str(DEFAULT_WS), "sep": os.sep, "os": platform.system()}})
            if u.path == "/api/test":
                return self.j(test_conn(q.get("url") or CFG["kobold"], q.get("key", "")))
            if u.path == "/api/rpc":
                return self.j(DISCORD.info())
            if u.path == "/api/wallpaper":
                if not WALL.exists():
                    return self.j({"error": "none"}, 404)
                return self.send_bytes(WALL.read_bytes(), "image/jpeg", extra={"Cache-Control": "max-age=31536000"})
            if u.path == "/api/ws/context":
                return self.j(ws_context(q["path"]))
            if u.path == "/api/ws/tree":
                return self.j({"files": ws_tree(q["path"])})
            if u.path == "/api/ws/file":
                _, p = safe(q["path"], q["rel"])
                if q.get("dl"):
                    return self.send_bytes(p.read_bytes(), "application/octet-stream",
                                           extra={"Content-Disposition": f'attachment; filename="{re.sub(r"[^A-Za-z0-9._-]", "_", p.name)}"'})
                try:
                    return self.j({"ok": True, "text": p.read_text("utf-8")[:400_000], "size": p.stat().st_size})
                except UnicodeDecodeError:
                    return self.j({"ok": False, "err": "binary file - use Download"})
            if u.path == "/api/ws/export":
                root, _ = safe(q["path"])
                name = re.sub(r"[^A-Za-z0-9._-]", "_", root.name or "project")
                return self.send_bytes(make_zip(root), "application/zip",
                                       extra={"Content-Disposition": f'attachment; filename="{name}.zip"'})
        except Exception as e:
            return self.j({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        self.j({"error": "not found"}, 404)

    def do_POST(self):
        if not self.authed():
            return self.j({"error": "unauthorized"}, 401)
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            p = self.path.split("?")[0]
            if p == "/api/state":
                tmp = STATE.with_suffix(".tmp")
                tmp.write_text(json.dumps(b), "utf-8")
                os.replace(tmp, STATE)
                return self.j({"ok": True})
            if p == "/api/generate":
                return self.generate(b)
            if p == "/api/abort":
                try:
                    kreq(b["url"], "/api/extra/abort", b.get("key", ""), {"genkey": b["genkey"]}, 3)
                except Exception:
                    pass
                return self.j({"ok": True})
            if p == "/api/tool":
                return self.j(run_tool(b["ws"], b["name"], b.get("args")))
            if p == "/api/rpc":
                return self.j(DISCORD.update(b))
            if p == "/api/wallpaper":
                if b.get("clear"):
                    WALL.unlink(missing_ok=True)
                    return self.j({"ok": True})
                raw = base64.b64decode(str(b.get("data", "")).split(",", 1)[-1])
                if not raw or len(raw) > 12_000_000:
                    return self.j({"ok": False, "error": "image missing or too large"}, 400)
                WALL.write_bytes(raw)
                return self.j({"ok": True, "v": int(time.time())})
            if p == "/api/undo":
                return self.j(undo(b["ws"]))
            if p == "/api/pickdir":
                return self.j({"path": pick_dir()})
            if p == "/api/open":
                open_folder(str(safe(b["path"])[0]))
                return self.j({"ok": True})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            return self.j({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        self.j({"error": "not found"}, 404)

    def generate(self, b):
        """Relay a streaming generation from KoboldCpp, normalised to  data: {"t": "<token>"}  events."""
        base, mode, key = b["url"].rstrip("/"), b.get("mode", "openai"), b.get("key", "")
        hdr = {"Content-Type": "application/json"}
        if key:
            hdr["Authorization"] = "Bearer " + key
        payload = b["payload"]
        if mode == "openai":
            url, payload["stream"] = base + "/v1/chat/completions", True
        else:
            url = base + "/api/extra/generate/stream"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(o):
            self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode())
            self.wfile.flush()

        try:
            req = urllib.request.Request(url, json.dumps(payload).encode(), hdr)
            with urllib.request.urlopen(req, timeout=900) as r:
                think = False
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                    except ValueError:
                        continue
                    if mode == "openai":
                        ch = (j.get("choices") or [{}])[0]
                        dl = ch.get("delta") or {}
                        rc, c, t = dl.get("reasoning_content") or dl.get("reasoning"), dl.get("content") or ch.get("text") or "", ""
                        if rc:
                            t += ("" if think else "<think>") + rc
                            think = True
                        if c:
                            t += ("</think>" if think else "") + c
                            think = False
                    else:
                        t = j.get("token") or ""
                    if t:
                        emit({"t": t})
            emit({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            if mode != "openai" or payload.get("genkey"):
                try:
                    kreq(base, "/api/extra/abort", key, {"genkey": payload.get("genkey", "")}, 3)
                except Exception:
                    pass
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode("utf-8", "replace")
            self.safe_emit(emit, {"err": f"KoboldCpp answered HTTP {e.code}: {body}"})
        except Exception as e:
            self.safe_emit(emit, {"err": f"Cannot reach KoboldCpp at {base}: {e}"})

    @staticmethod
    def safe_emit(emit, o):
        try:
            emit(o)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="KoboldCode - Claude-style chat + coding agent for KoboldCpp")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--kobold", default="http://localhost:5001", help="default KoboldCpp URL")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", action="store_true", help="open in a native window (needs pywebview)")
    a = ap.parse_args()
    CFG.update(kobold=a.kobold, host=a.host)
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer((a.host, port), Handler)
            break
        except OSError:
            continue
    if not srv:
        sys.exit("No free port found.")
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{srv.server_port}/"
    print(f"\n  KoboldCode  ->  {url}\n  KoboldCpp   ->  {a.kobold}\n  Data        ->  {APP}\n  Ctrl+C to quit\n")
    if a.window:
        try:
            import webview
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            webview.create_window("KoboldCode", url, width=1280, height=820)
            webview.start()
            return
        except ImportError:
            print("  (pywebview not installed - opening your browser instead: pip install pywebview)\n")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
