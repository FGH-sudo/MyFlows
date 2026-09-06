"""Synchronous parameter owner and the only process applying optimizer updates."""

import time
import traceback

from .model import SmallMLP
from .protocol import aggregate, clone, event, payload_bytes, payload_hash, receive, schema_hash, send, validate_gradient


def server_main(config, incoming, replies, events, results, stop):
    model = SmallMLP(config["seed"])
    history, losses = [model.snapshot()], []
    step = -1
    try:
        for step in range(config["steps"] + 1):
            step_start = time.perf_counter()
            params = model.snapshot()
            parameter_hash = payload_hash(params)
            for reply in replies:
                send(reply, {"kind": "parameters", "run_id": config["run_id"], "step_id": step,
                             "parameter_version": step, "parameters": clone(params),
                             "schema_hash": schema_hash(params), "parameter_hash": parameter_hash}, stop, config["timeout"])
            broadcast_s = time.perf_counter() - step_start
            received = {}
            collect_start = time.perf_counter()
            while len(received) < config["workers"]:
                message = receive(incoming, stop, config["timeout"])
                if step == config["steps"]:
                    worker = message.get("worker_id")
                    expected = {"kind": "done", "run_id": config["run_id"], "worker_id": worker,
                                "parameter_version": step, "parameter_hash": parameter_hash}
                    if message != expected or type(worker) is not int or not 0 <= worker < config["workers"] or worker in received:
                        raise ValueError("invalid/duplicate final acknowledgement")
                else:
                    worker = validate_gradient(message, run_id=config["run_id"], step=step,
                                               parameter_hash=parameter_hash, reference=params,
                                               shard_sizes=config["shard_sizes"], received=received)
                    send(replies[worker], {"kind": "ack", "run_id": config["run_id"],
                                          "worker_id": worker, "step_id": step}, stop, config["timeout"])
                received[worker] = message
            collect_s = time.perf_counter() - collect_start
            if step == config["steps"]:
                break
            aggregate_start = time.perf_counter()
            gradients, loss, total = aggregate([received[i] for i in range(config["workers"])])
            aggregate_s = time.perf_counter() - aggregate_start
            update_start = time.perf_counter()
            model.update(gradients)
            update_s = time.perf_counter() - update_start
            history.append(model.snapshot())
            losses.append(loss)
            event(events, stop, config, "server", None, step, "updated", parameter_version=step + 1,
                  n_samples=total, loss=loss, collect_wait_s=collect_s, aggregate_s=aggregate_s,
                  update_s=update_s, broadcast_submit_s=broadcast_s, step_s=time.perf_counter() - step_start,
                  parameter_payload_bytes=payload_bytes(params) * config["workers"],
                  gradient_payload_bytes=sum(payload_bytes(m["gradients"]) for m in received.values()))
        send(results, {"status": "passed", "history": history, "losses": losses}, stop, config["timeout"])
    except BaseException as exc:
        # The result channel is drained by the launcher even after a stop signal.
        results.put({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                     "failed_step": step, "traceback": traceback.format_exc()}, timeout=1)
        stop.set()
        raise SystemExit(1) from None
