# AIperf Benchmark Playbook

This folder holds the four benchmark runs (aggregation vs disaggregation+router, workloads A and B) for Qwen/Qwen3-0.6B using NVIDIA Dynamo. Runs can target either a vLLM-based router or a TRT-LLM-based router; pick the right port forward before starting disaggregation+router tests (see below).

## Shared flags (use in every run)
```
--model Qwen/Qwen3-0.6B --url http://localhost:8000 --endpoint-type chat --endpoint /v1/chat/completions --streaming --warmup-request-count 40 --extra-inputs ignore_eos:true
```

## Backend and port forwarding (disaggregation+router)
Keep port 8000 forwarded to the router frontend that matches the inference backend you are testing:
- **vLLM backend:** `kubectl port-forward service/llm-disagg-router-frontend-app 8000:8000 -n dynamo-cloud`
- **TRT-LLM backend:** `kubectl port-forward service/trtllm-disagg-router-frontend-app 8000:8000 -n dynamo-cloud`

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
3. Switch backend to **Disaggregation + Router** (same model/config) and ensure the matching port forward above is running.
4. Rerun Workload A and Workload B.
5. Move all the generated folders and files from artifacts to artifacts_benchmark
6. Compare metrics. Expect clearer gains in Workload B (decode-heavy) and potential wins in Workload A if prefill splitting reduces queuing.

## Worker balancing flag
Disagg folders now always encode prefill/decode counts and concurrency, e.g., `...-workload-A-3P-1D-24C`.
- Naming rule: `...-workload-<A|B>-<P>P-<D>D-<C>C`, where `P` is the number of prefill workers, `D` is the number of decode workers, and `C` is the concurrency used for that run.
- `--worker-balancing` (default **off**) means: only pick 2P-2D disagg folders for each workload.
- Without the flag: only pick non-2P-2D disagg folders for each workload.
If zero or multiple folders match for a workload, the script errors so you can disambiguate.

## Plotting
Generate the two-panel throughput + TTFT chart (Workload A and B) from the four runs:
```
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend vllm --concurrency 24
# Use 2P-2D disagg folders
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend vllm --concurrency 24 --worker-balancing

# TRT-LLM examples (adjust concurrency to your folders)
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend trtllm --concurrency 96
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend trtllm --concurrency 96 --worker-balancing
```
Pass `--concurrency` to match the `*C` suffix in your folder names. The script expects four folders:
- Agg: `<model>-agg-<backend>-workload-{A|B}-<C>c`
- Disagg: `<model>-disagg-router-<backend>-workload-{A|B}-<P>P-<D>D-<C>C` (filtered by the worker-balancing rule above)

Outputs: `artifacts_benchmark/output_token_throughput_<model>_<backend>_c<concurrency>.png` plus a table of throughput and TTFT (with WARN flags if errors were recorded).
