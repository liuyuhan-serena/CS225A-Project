import argparse
import ast
import json
import signal
import time

import numpy as np
import redis


RAW_TARGET_KEY = "opensai::perception::desired_position"
SMOOTH_TARGET_KEY = "opensai::perception::desired_position_smoothed"
SMOOTH_VELOCITY_KEY = "opensai::perception::desired_velocity_smoothed"
SMOOTH_METADATA_KEY = "opensai::perception::smooth_target_metadata"


def parse_vector(raw):
    if raw is None:
        return None

    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(value, list) or len(value) != 3:
        return None

    try:
        vector = np.array(value, dtype=float)
    except ValueError:
        return None

    if not np.all(np.isfinite(vector)):
        return None

    return vector


def vector_string(vector):
    return json.dumps([float(x) for x in vector], separators=(",", ":"))


def min_jerk(s):
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def limit_step(previous, desired, max_speed, dt):
    delta = desired - previous
    distance = float(np.linalg.norm(delta))
    max_step = max_speed * dt

    if distance <= max_step or distance <= 1e-12:
        return desired

    return previous + delta * (max_step / distance)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth a raw Redis perception target into a controller-friendly target."
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--input-key", default=RAW_TARGET_KEY)
    parser.add_argument("--output-key", default=SMOOTH_TARGET_KEY)
    parser.add_argument("--velocity-key", default=SMOOTH_VELOCITY_KEY)
    parser.add_argument("--metadata-key", default=SMOOTH_METADATA_KEY)
    parser.add_argument("--rate-hz", type=float, default=1000.0)
    parser.add_argument("--input-poll-hz", type=float, default=50.0)
    parser.add_argument("--chunk-duration", type=float, default=0.35)
    parser.add_argument("--ema-tau", type=float, default=0.08)
    parser.add_argument("--max-speed", type=float, default=0.20)
    parser.add_argument("--target-change-threshold", type=float, default=0.005)
    parser.add_argument("--stale-timeout", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()
    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        decode_responses=True
    )
    redis_client.ping()

    running = True

    def stop(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    dt_target = 1.0 / args.rate_hz
    raw_target = None
    active_target = None
    smoothed = None
    previous_smoothed = None
    chunk_start = None
    chunk_goal = None
    chunk_start_time = None
    last_valid_raw_time = None
    seq = 0
    next_tick = time.perf_counter()
    last_tick = next_tick
    next_input_poll = next_tick
    input_poll_period = 1.0 / args.input_poll_hz if args.input_poll_hz > 0 else 0.0

    print("Smoothing Redis target:")
    print(f"  input:    {args.input_key}")
    print(f"  output:   {args.output_key}")
    print(f"  velocity: {args.velocity_key}")
    print(f"  metadata: {args.metadata_key}")
    print("  Ctrl-C to quit")

    while running:
        now_perf = time.perf_counter()

        if now_perf < next_tick:
            time.sleep(min(next_tick - now_perf, 0.001))
            continue

        now = time.time()
        dt = max(now_perf - last_tick, 1e-6)
        last_tick = now_perf
        next_tick += dt_target

        polled_raw_valid = False

        if now_perf >= next_input_poll:
            next_input_poll = now_perf + input_poll_period
            candidate = parse_vector(redis_client.get(args.input_key))
            polled_raw_valid = candidate is not None

            if polled_raw_valid:
                last_valid_raw_time = now

                if raw_target is None:
                    raw_target = candidate
                elif np.linalg.norm(candidate - raw_target) >= args.target_change_threshold:
                    raw_target = candidate

        raw_is_stale = (
            last_valid_raw_time is None
            or now - last_valid_raw_time > args.stale_timeout
        )

        if raw_target is not None and not raw_is_stale:
            if active_target is None:
                active_target = raw_target.copy()
                smoothed = raw_target.copy()
                previous_smoothed = raw_target.copy()
                chunk_start = raw_target.copy()
                chunk_goal = raw_target.copy()
                chunk_start_time = now
            elif np.linalg.norm(raw_target - active_target) >= args.target_change_threshold:
                active_target = raw_target.copy()
                chunk_start = smoothed.copy()
                chunk_goal = active_target.copy()
                chunk_start_time = now

        valid_output = smoothed is not None and not raw_is_stale

        if valid_output:
            if chunk_start is None or chunk_goal is None or chunk_start_time is None:
                chunk_desired = smoothed
            else:
                s = (now - chunk_start_time) / max(args.chunk_duration, 1e-6)
                blend = min_jerk(s)
                chunk_desired = chunk_start + blend * (chunk_goal - chunk_start)

            alpha = 1.0 - np.exp(-dt / max(args.ema_tau, 1e-6))
            ema_desired = smoothed + alpha * (chunk_desired - smoothed)
            smoothed = limit_step(smoothed, ema_desired, args.max_speed, dt)
            velocity = (smoothed - previous_smoothed) / dt
            previous_smoothed = smoothed.copy()
            output_value = vector_string(smoothed)
            velocity_value = vector_string(velocity)
        else:
            output_value = "[]"
            velocity_value = "[]"

        seq += 1
        metadata = {
            "seq": seq,
            "timestamp": now,
            "valid": valid_output,
            "polled_raw_valid": polled_raw_valid,
            "raw_stale": raw_is_stale,
            "rate_hz": args.rate_hz,
            "input_poll_hz": args.input_poll_hz,
            "chunk_duration": args.chunk_duration,
            "ema_tau": args.ema_tau,
            "max_speed": args.max_speed,
            "input_key": args.input_key,
            "output_key": args.output_key,
        }

        pipe = redis_client.pipeline(transaction=True)
        pipe.set(args.output_key, output_value)
        pipe.set(args.velocity_key, velocity_value)
        pipe.set(args.metadata_key, json.dumps(metadata, separators=(",", ":")))
        pipe.execute()

    print("\nStopped.")


if __name__ == "__main__":
    main()
