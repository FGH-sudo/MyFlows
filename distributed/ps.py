"""PS process entry: serve the shared engine over Socket or gRPC."""

from __future__ import annotations

import traceback
import threading

from .constants import (
    BIND_HOST,
    DEFAULT_PS_DIGEST_ROUNDS,
    DEFAULT_PS_RETAIN_ROUNDS,
)
from .engine import PSEngine
from .monitor import emit
from .session import TrainSession
from .shards import split_global_batch
from .transport_socket import serve_socket


def build_engine(config):
    snapshot = TrainSession(config, for_compute=False)
    sizes = split_global_batch(int(config["global_batch"]), int(config["train_workers"]),
                               config.get("shard_sizes"))
    engine = PSEngine(
        run_id=config["run_id"],
        n_workers=int(config["train_workers"]),
        init_parameters=snapshot.named_parameters_cpu(),
        optimizer_meta=snapshot.optimizer_meta(),
        shard_sizes=sizes,
        heartbeat_timeout_s=float(config.get("heartbeat_timeout_s", 5.0)),
        reconnect_wait_s=float(config.get("reconnect_wait_s", 10.0)),
        retain_rounds=int(config.get("retain_rounds", DEFAULT_PS_RETAIN_ROUNDS)),
        digest_rounds=int(config.get("digest_rounds", DEFAULT_PS_DIGEST_ROUNDS)),
        wait_strategy=config.get("ps_wait_strategy", "poll"),
    )
    engine.committed_version = snapshot.local_version
    engine.initial_version = snapshot.local_version
    return engine, snapshot


def server_main(config, events, results, stop):
    engine = None
    try:
        engine, snapshot = build_engine(config)
        def watchdog():
            while not stop.wait(0.1):
                missing = engine.check_liveness()
                if missing:
                    emit(events, config, 'server', None, -1, 'heartbeat_timeout', missing=missing)
                    stop.set()
                    return
        threading.Thread(target=watchdog, daemon=True).start()
        emit(events, config, "server", None, -1, "listening",
             port=config.get("ps_port"), transport=config.get("transport"))
        transport = config.get("transport", "socket_json")
        if transport == "socket_json":
            serve_socket(engine, config["ps_port"], stop, host=config.get("bind_host", BIND_HOST),
                         timeout=float(config.get("timeout", 30)))
        elif transport == "grpc_proto":
            from .transport_grpc import serve_ps_grpc
            serve_ps_grpc(engine, config["ps_port"], stop, timeout=float(config.get("timeout", 30)))
        else:
            raise ValueError(f"unsupported PS transport {transport}")
        if engine.aborted:
            raise RuntimeError(engine.aborted)
        emit(events, config, "server", None, -1, "cache_stats",
             cache_sizes=engine.cache_sizes(),
             cache_footprint_bytes=engine.cache_footprint_bytes(),
             retained_payload_bytes=engine.retained_payload_bytes())
        try:
            results.put({
                "status": "passed" if not engine.aborted else "failed",
                "error": engine.aborted,
                "init_parameter_hash": snapshot.parameter_hash(),
                "schema_hash": snapshot.schema_hash(),
            }, timeout=1)
        except Exception:
            pass
    except BaseException as exc:
        if engine is not None:
            engine.abort(str(exc))
        try:
            results.put({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }, timeout=1)
        except Exception:
            pass
        stop.set()
        raise SystemExit(1) from None
