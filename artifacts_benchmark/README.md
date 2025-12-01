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
    ```

    ```

## Worker balancing flag
If your disaggregation runs use explicit prefill/decode worker splits in the folder name (e.g., `...-workload-A-3P-1D` meaning 3 prefill workers and 1 decode worker), add `--worker-balancing` so the plotter picks up those directories. Without the flag, it uses the default `...-workload-A`/`...-workload-B` disagg folders (2 prefill + 2 decode).

## Plotting
Generate the two-panel throughput + TTFT chart (Workload A and B) from the four runs:
```
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend vllm
# Add --worker-balancing if using the ...-<n>P-<m>D disagg directories
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend vllm --worker-balancing

# When using TRT-LLM as the inference backend
python artifacts_benchmark/plot_benchmarks.py --model "Qwen/Qwen3-0.6B" --inference-backend trtllm
```
This saves `artifacts_benchmark/output_token_throughput_<model>_<backend>.png` and prints a small table of throughput and TTFT (with WARN flags if errors were recorded).
