from __future__ import annotations

import base64
import json
import subprocess
import time

import numpy as np
import requests
import cv2
import websocket

from .base import FrameResult, FrameSource


class HeadlessBrowserSource(FrameSource):
    source_type = "browser_headless"

    def __init__(self, url: str, chrome_path: str = r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 port: int = 0, width: int = 1280, height: int = 720, wait_sec: float = 10.0):
        self.url = url
        self.chrome_path = chrome_path
        self.port = port if port > 0 else self._find_free_port()
        self.width = width
        self.height = height
        self.wait_sec = wait_sec
        self._proc = None
        self._ws = None
        self._msg_id = 0
        self._frame_idx = 0

    @staticmethod
    def _find_free_port() -> int:
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def open(self) -> bool:
        if not self.url:
            return False
        import tempfile, os

        user_data = os.path.join(tempfile.gettempdir(), f"chrome_headless_{self.port}")
        try:
            self._proc = subprocess.Popen([
                self.chrome_path,
                "--headless=new", "--disable-gpu", "--no-sandbox",
                f"--remote-debugging-port={self.port}",
                f"--window-size={self.width},{self.height}",
                "--mute-audio", "--autoplay-policy=no-user-gesture-required",
                "--remote-allow-origins=*",
                f"--user-data-dir={user_data}",
                self.url,
            ])
        except Exception:
            return False
        for _ in range(int(self.wait_sec * 2)):
            if self._proc.poll() is not None:
                return False
            try:
                tabs = requests.get(f"http://127.0.0.1:{self.port}/json/list", timeout=2).json()
                ws_url = None
                for t in tabs:
                    if t.get("type") == "page":
                        ws_url = t.get("webSocketDebuggerUrl")
                        break
                if ws_url:
                    self._ws = websocket.create_connection(ws_url, timeout=10)
                    self._frame_idx = 0
                    time.sleep(2)
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        self.close()
        return False

    def _send_cdp(self, method: str, params: dict | None = None) -> dict:
        self._msg_id += 1
        msg = {"id": self._msg_id, "method": method}
        if params:
            msg["params"] = params
        self._ws.send(json.dumps(msg))
        while True:
            resp = json.loads(self._ws.recv())
            if resp.get("id") == self._msg_id:
                return resp

    def _grab_video_frame(self) -> np.ndarray | None:
        js = """
        (function() {
            var v = document.querySelector('video');
            if (!v || v.videoWidth == 0) return null;
            var c = document.createElement('canvas');
            c.width = v.videoWidth;
            c.height = v.videoHeight;
            c.getContext('2d').drawImage(v, 0, 0);
            return c.toDataURL('image/jpeg', 0.8).split(',')[1];
        })();
        """
        try:
            resp = self._send_cdp("Runtime.evaluate", {
                "expression": js,
                "returnByValue": True,
            })
            val = resp.get("result", {}).get("result", {}).get("value")
            if not val:
                return None
            img_bytes = base64.b64decode(val)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _grab_screenshot(self) -> np.ndarray | None:
        try:
            resp = self._send_cdp("Page.captureScreenshot", {"format": "jpeg", "quality": 80})
            data = resp.get("result", {}).get("data")
            if not data:
                return None
            img_bytes = base64.b64decode(data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def next_frame(self) -> FrameResult:
        if self._ws is None:
            return FrameResult.failure("浏览器未打开", self.source_type)
        frame = self._grab_video_frame()
        if frame is None:
            frame = self._grab_screenshot()
        if frame is None:
            return FrameResult.failure("无法抓取画面", self.source_type, self._frame_idx)
        res = FrameResult.success(frame, self._frame_idx, self.source_type)
        self._frame_idx += 1
        return res

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
