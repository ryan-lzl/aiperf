# AIperf Benchmark Playbook

This folder holds the four benchmark runs (aggregation vs disaggregation+router, workloads A and B) for Qwen/Qwen3-0.6B using NVIDIA Dynamo.

## Shared flags (use in every run)
```
--model Qwen/Qwen3-0.6B --url http://localhost:8000 --endpoint-type chat --endpoint /v1/chat/completions --streaming --warmup-request-count 40 --extra-inputs ignore_eos:true
```

## Workloads
- **Workload A (prefill-heavy, short decode):** highlights whether prefill disaggregation helps  
  ```
  aiperf profile $SHARED --concurrency 24 --request-count 600 --synthetic-input-tokens-mean 900 --synthetic-input-tokens-stddev 200 --output-tokens-mean 160 --output-tokens-stddev 40 --extra-inputs min_tokens:160
  ```
- **Workload B (decode-heavy):** stresses decode and router load-balancing/speculation  
  ```
  aiperf profile $SHARED --concurrency 24 --request-count 600 --synthetic-input-tokens-mean 180 --synthetic-input-tokens-stddev 50 --output-tokens-mean 600 --output-tokens-stddev 120 --extra-inputs min_tokens:500
  ```

## Procedure
1. Start backend in **Aggregation** mode and warm it.
2. Run Workload A, then Workload B. Record tokens/s, TTFT, p95/p99.
3. Switch backend to **Disaggregation + Router** (same model/config).
4. Rerun Workload A and Workload B.
5. Compare metrics. Expect clearer gains in Workload B (decode-heavy) and potential wins in Workload A if prefill splitting reduces queuing.

Optional: if the client supports open-loop arrivals, enable it to avoid closed-loop backpressure masking backend differences.

## Plotting
Generate the two-panel throughput + TTFT chart (Workload A and B) from the four runs:
```
python artifacts_benchmark/plot_benchmarks.py
```
This saves `artifacts_benchmark/output_token_throughput.png` and prints a small table of throughput and TTFT (with WARN flags if errors were recorded).
