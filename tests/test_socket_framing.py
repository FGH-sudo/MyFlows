import socket
import struct
import threading
import time
import unittest

from MyFlows.distributed.protocol import pack_frame
from MyFlows.distributed.transport_socket import ConnectionBuffer, FrameDeadlineExceeded, serve_socket


class _ChunkSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.timeout = None

    def recv(self, _n):
        if not self.chunks:
            raise socket.timeout()
        chunk = self.chunks.pop(0)
        if chunk is None:
            raise socket.timeout()
        return chunk

    def settimeout(self, value):
        self.timeout = value


class _DummyEngine:
    stopped = False

    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()

    def handle(self, message, wait_s=0.0):
        with self.lock:
            self.messages.append(message)
        return {"status": "OK", "echo": message.get("n"), "request_id": message.get("request_id")}


class SocketFramingTest(unittest.TestCase):
    def test_complete_frame(self):
        payload = pack_frame({"n": 1, "type": "push"})
        buf = ConnectionBuffer(_ChunkSocket([payload]), max_bytes=1024)
        self.assertEqual(buf.read_message(deadline=time.monotonic() + 1), {"n": 1, "type": "push"})

    def test_one_to_three_byte_chunks(self):
        payload = pack_frame({"n": 7, "ok": True})
        chunks = [payload[i:i + 2] for i in range(0, len(payload), 2)]
        buf = ConnectionBuffer(_ChunkSocket(chunks), max_bytes=1024)
        self.assertEqual(buf.read_message(deadline=time.monotonic() + 1)["n"], 7)

    def test_two_sticky_frames_in_one_recv(self):
        first = pack_frame({"n": 1})
        second = pack_frame({"n": 2})
        buf = ConnectionBuffer(_ChunkSocket([first + second]), max_bytes=1024)
        self.assertEqual(buf.read_message(deadline=time.monotonic() + 1)["n"], 1)
        self.assertEqual(buf.read_message(deadline=time.monotonic() + 1)["n"], 2)

    def test_timeout_keeps_partial_frame_then_completes(self):
        payload = pack_frame({"n": 9, "body": "abcdef"})
        partial = payload[:6]
        rest = payload[6:]
        sock = _ChunkSocket([partial, None, rest])
        buf = ConnectionBuffer(sock, max_bytes=1024)
        deadline = time.monotonic() + 1
        with self.assertRaises(socket.timeout):
            buf.read_message(deadline=deadline)
        self.assertGreater(len(buf.buf), 0)
        self.assertEqual(buf.read_message(deadline=time.monotonic() + 1)["n"], 9)

    def test_oversize_frame_rejected_from_header(self):
        header = struct.pack("!I", 64)
        sock = _ChunkSocket([header])
        buf = ConnectionBuffer(sock, max_bytes=8)
        with self.assertRaises(ValueError):
            buf.read_message(deadline=time.monotonic() + 1)

    def _serve(self, timeout, idle_timeout):
        engine = _DummyEngine()
        stop = threading.Event()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.close()
        thread = threading.Thread(
            target=serve_socket, args=(engine, port, stop),
            kwargs={"host": "127.0.0.1", "timeout": timeout, "idle_timeout": idle_timeout, "max_bytes": 4096},
            daemon=True,
        )
        thread.start()
        return engine, stop, thread, port

    def _connect(self, port):
        deadline = time.time() + 2
        while True:
            try:
                return socket.create_connection(("127.0.0.1", port), timeout=0.2)
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.02)

    def _recv_reply(self, client, timeout=1.0):
        client.settimeout(timeout)
        header = b""
        while len(header) < 4:
            chunk = client.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        length = struct.unpack("!I", header)[0]
        body = b""
        while len(body) < length:
            chunk = client.recv(length - len(body))
            if not chunk:
                return None
            body += chunk
        return header + body

    def test_serve_socket_survives_timeout_mid_frame(self):
        engine, stop, thread, port = self._serve(timeout=2.0, idle_timeout=0.2)
        try:
            client = self._connect(port)
            payload = pack_frame({"n": 42, "request_id": "mid-timeout"})
            client.sendall(payload[:5])
            time.sleep(0.55)
            client.sendall(payload[5:])
            reply = self._recv_reply(client, timeout=1.5)
            client.close()
            self.assertIsNotNone(reply)
            self.assertEqual(engine.messages[-1]["n"], 42)
        finally:
            stop.set()
            thread.join(timeout=2)

    def test_partial_frame_past_deadline_is_rejected(self):
        engine, stop, thread, port = self._serve(timeout=0.25, idle_timeout=0.08)
        try:
            client = self._connect(port)
            payload = pack_frame({"n": 7, "request_id": "too-slow"})
            client.sendall(payload[:5])
            time.sleep(0.85)
            try:
                client.sendall(payload[5:])
                reply = self._recv_reply(client, timeout=0.5)
            except OSError:
                reply = None
            client.close()
            self.assertFalse(engine.messages)
            self.assertIsNone(reply)
        finally:
            stop.set()
            thread.join(timeout=2)

    def test_idle_connection_can_receive_a_new_frame(self):
        engine, stop, thread, port = self._serve(timeout=1.0, idle_timeout=0.15)
        try:
            client = self._connect(port)
            time.sleep(0.4)
            payload = pack_frame({"n": 3, "request_id": "after-idle"})
            client.sendall(payload)
            reply = self._recv_reply(client, timeout=1.0)
            client.close()
            self.assertIsNotNone(reply)
            self.assertEqual(engine.messages[-1]["n"], 3)
        finally:
            stop.set()
            thread.join(timeout=2)

    def test_trickle_cannot_refresh_frame_deadline(self):
        engine, stop, thread, port = self._serve(timeout=0.3, idle_timeout=0.08)
        try:
            client = self._connect(port)
            payload = pack_frame({"n": 99, "request_id": "trickle", "pad": "x" * 40})
            for i in range(0, len(payload), 2):
                try:
                    client.sendall(payload[i:i + 2])
                except OSError:
                    break
                time.sleep(0.06)
            try:
                reply = self._recv_reply(client, timeout=0.5)
            except OSError:
                reply = None
            client.close()
            self.assertFalse(engine.messages)
            self.assertIsNone(reply)
        finally:
            stop.set()
            thread.join(timeout=2)

    def test_connection_buffer_raises_frame_deadline(self):
        payload = pack_frame({"n": 1})
        sock = _ChunkSocket([payload[:3]] + [None] * 100)
        buf = ConnectionBuffer(sock, max_bytes=1024)
        deadline = time.monotonic() + 0.05
        with self.assertRaises(FrameDeadlineExceeded):
            buf.read_message(deadline=deadline, idle_timeout=0.01)


if __name__ == "__main__":
    unittest.main()
