"""Spawn-safe CPU worker. Workers calculate gradients but never update weights."""

import os
import time

from .model import SmallMLP, make_batches
from .protocol import clone, event, payload_bytes, payload_hash, receive, schema_hash, send, validate_arrays


def worker_main(worker_id, config, shard, incoming, gradients, events, stop):
    model = SmallMLP(config["seed"])
    reference = model.snapshot()
    x, y = make_batches(config["seed"], config["global_batch"], config["steps"])
    try:
        for step in range(config["steps"] + 1):
            waited = time.perf_counter()
            message = receive(incoming, stop, config["timeout"])
            if (message.get("kind"), message.get("run_id"), message.get("step_id"), message.get("parameter_version")) != (
                    "parameters", config["run_id"], step, step):
                raise ValueError("invalid parameter version/message")
            params = message["parameters"]
            validate_arrays(params, reference)
            parameter_hash = payload_hash(params)
            if message["schema_hash"] != schema_hash(reference) or message["parameter_hash"] != parameter_hash:
                raise ValueError("invalid parameter schema/hash")
            model.load(params)
            event(events, stop, config, "worker", worker_id, step, "parameters_received",
                  parameter_version=step, parameter_wait_receive_s=time.perf_counter() - waited,
                  parameter_payload_bytes=payload_bytes(params))
            if step == config["steps"]:
                send(gradients, {"kind": "done", "run_id": config["run_id"], "worker_id": worker_id,
                                 "parameter_version": step, "parameter_hash": parameter_hash}, stop, config["timeout"])
                return
            fault = config.get("fault") if worker_id == 0 and step == config.get("fault_step", 0) else None
            if fault == "crash":
                os._exit(17)
            if fault == "timeout":
                stop.wait(config["timeout"] * 3)
                raise TimeoutError("injected worker timeout")
            started = time.perf_counter()
            loss, values = model.gradients(x[step, shard], y[step, shard])
            compute_s = time.perf_counter() - started
            message = {"kind": "gradient", "run_id": config["run_id"], "worker_id": worker_id,
                       "step_id": step, "parameter_version": step, "parameter_hash": parameter_hash,
                       "schema_hash": schema_hash(reference), "n_samples": len(shard),
                       "loss": loss, "gradients": clone(values), "gradient_hash": payload_hash(values)}
            if fault == "schema":
                message["schema_hash"] = "injected-invalid-schema"
            submit_start = time.perf_counter()
            send(gradients, message, stop, config["timeout"])
            if fault == "duplicate":
                send(gradients, message, stop, config["timeout"])
            submitted = time.perf_counter()
            ack = receive(incoming, stop, config["timeout"])
            if ack != {"kind": "ack", "run_id": config["run_id"], "worker_id": worker_id, "step_id": step}:
                raise ValueError("invalid upload acknowledgement")
            event(events, stop, config, "worker", worker_id, step, "gradient_acknowledged",
                  parameter_version=step, n_samples=len(shard), compute_s=compute_s,
                  upload_submit_s=submitted - submit_start, upload_ack_wait_s=time.perf_counter() - submitted,
                  gradient_payload_bytes=payload_bytes(values))
    except BaseException as exc:
        if not stop.is_set():
            event(events, stop, config, "worker", worker_id, locals().get("step", -1), "error", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
