"""Length-prefixed JSON sockets bound to 127.0.0.1."""

from __future__ import annotations

import socket
import struct
import threading
import time

from .constants import BIND_HOST, DEFAULT_MAX_FRAME, DEFAULT_REQUEST_TIMEOUT_S
from .protocol import pack_frame, recv_frame_from_buffer, unpack_frame


class FrameDeadlineExceeded(TimeoutError):
    """Absolute deadline for one in-progress frame elapsed."""


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed during recv")
        buf.extend(chunk)
    return bytes(buf)


def send_message(sock, obj, max_bytes=DEFAULT_MAX_FRAME):
    sock.sendall(pack_frame(obj, max_bytes=max_bytes))


def recv_message(sock, max_bytes=DEFAULT_MAX_FRAME):
    header = recv_exact(sock, 4)
    length = struct.unpack("!I", header)[0]
    if length > int(max_bytes):
        raise ValueError(f"frame length {length} exceeds {max_bytes}")
    body = recv_exact(sock, length)
    return unpack_frame(header + body)


class ConnectionBuffer:
    """Per-connection receive buffer. Timeouts never discard a partial frame."""

    def __init__(self, sock, max_bytes=DEFAULT_MAX_FRAME):
        self.sock = sock
        self.max_bytes = int(max_bytes)
        self.buf = bytearray()
        self._frame_deadline = None

    def read_message(self, deadline=None, idle_timeout=None, frame_timeout=None):
        while True:
            frame, self.buf = recv_frame_from_buffer(self.buf, self.max_bytes)
            if frame is not None:
                self.last_frame_bytes = len(frame)
                self._frame_deadline = None
                return unpack_frame(frame)
            now = time.monotonic()
            if self.buf:
                if self._frame_deadline is None:
                    if deadline is not None:
                        self._frame_deadline = deadline
                    elif frame_timeout is not None:
                        self._frame_deadline = now + float(frame_timeout)
            if self._frame_deadline is not None and now >= self._frame_deadline:
                raise FrameDeadlineExceeded("frame deadline exceeded")
            if deadline is not None and not self.buf and now >= deadline:
                raise FrameDeadlineExceeded("request deadline exceeded")
            timeout = idle_timeout
            cap = self._frame_deadline if self._frame_deadline is not None else deadline
            if cap is not None:
                remain = max(0.0, cap - time.monotonic())
                if remain <= 0:
                    raise FrameDeadlineExceeded("frame deadline exceeded")
                timeout = remain if timeout is None else min(float(timeout), remain)
            if timeout is not None:
                self.sock.settimeout(timeout if timeout > 0 else 0.001)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                now = time.monotonic()
                if self._frame_deadline is not None and now >= self._frame_deadline:
                    raise FrameDeadlineExceeded("frame deadline exceeded") from None
                if deadline is not None and now >= deadline:
                    raise FrameDeadlineExceeded("request deadline exceeded") from None
                if self.buf and idle_timeout is not None:
                    continue
                raise
            if not chunk:
                raise ConnectionError("socket closed during recv")
            self.buf.extend(chunk)
            if self._frame_deadline is None:
                now = time.monotonic()
                if deadline is not None:
                    self._frame_deadline = deadline
                elif frame_timeout is not None:
                    self._frame_deadline = now + float(frame_timeout)


class SocketClient:
    def __init__(self, host, port, timeout=DEFAULT_REQUEST_TIMEOUT_S, max_bytes=DEFAULT_MAX_FRAME):
        self.max_bytes = max_bytes
        self.timeout = float(timeout)
        self.lock = threading.Lock()
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)
        self.recvbuf = ConnectionBuffer(self.sock, max_bytes)
        self.traffic = {}

    def call(self, message):
        with self.lock:
            from .measurement import account
            started = time.perf_counter()
            frame = pack_frame(message, self.max_bytes)
            encode_s = time.perf_counter() - started
            started = time.perf_counter()
            self.sock.sendall(frame)
            deadline = time.monotonic() + self.timeout
            reply = self.recvbuf.read_message(deadline=deadline, idle_timeout=self.timeout)
            account(self, message.get('type'), len(frame)-4, self.recvbuf.last_frame_bytes-4,
                    time.perf_counter()-started, encode_s,
                    payload=sum(4*len(t['data']) for t in message.get('tensors', [])))
            self.traffic['socket_framing_bytes'] = self.traffic.get('socket_framing_bytes', 0) + 8
            return reply

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def serve_socket(
    engine, port, stop, host=BIND_HOST, max_bytes=DEFAULT_MAX_FRAME,
    timeout=DEFAULT_REQUEST_TIMEOUT_S, idle_timeout=None,
):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(64)
    server.settimeout(0.2)
    threads = []
    request_timeout = float(timeout)
    idle = float(idle_timeout) if idle_timeout is not None else min(0.5, request_timeout)

    def handle(conn):
        recvbuf = ConnectionBuffer(conn, max_bytes)
        conn.settimeout(idle)
        try:
            while not stop.is_set() and not engine.stopped:
                try:
                    message = recvbuf.read_message(
                        idle_timeout=idle, frame_timeout=request_timeout)
                except FrameDeadlineExceeded:
                    break
                except socket.timeout:
                    if stop.is_set():
                        break
                    continue
                except TimeoutError as exc:
                    if isinstance(exc, FrameDeadlineExceeded):
                        break
                    if stop.is_set():
                        break
                    continue
                except (ConnectionError, OSError, ValueError):
                    break
                reply = engine.handle(message, wait_s=0.0)
                try:
                    send_message(conn, reply, max_bytes)
                except OSError:
                    break
        finally:
            conn.close()

    try:
        while not stop.is_set() and not engine.stopped:
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=handle, args=(conn,), daemon=True)
            thread.start()
            threads.append(thread)
    finally:
        server.close()
        deadline = time.monotonic() + 2
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
