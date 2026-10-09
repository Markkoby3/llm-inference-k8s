# Design decisions

Short records of the choices that shape this project, with the alternatives
considered.

## 1. A thin gateway in front of interchangeable engines

**Decision.** Clients talk to one OpenAI-compatible endpoint. A backend adapter
interface (`complete`, `stream`, `ready`) hides each engine's API, and the
`X-InferScale-Backend` header picks the engine per request.

**Why.** Swapping or A/B-testing an engine becomes a deploy-time setting, not a
client change. Adding TensorRT-LLM or SGLang means writing one adapter.

**Cost.** One extra network hop. The harness can target vLLM directly
(`--url http://vllm:8000`) to measure exactly what that hop costs.

## 2. Compare serving layers with the same engine underneath

**Decision.** Triton runs NVIDIA's vLLM backend, so both paths execute the same
vLLM engine on the same weights.

**Why.** Comparing vLLM with Triton + TensorRT-LLM would mix two effects: kernel
and engine differences, and serving-layer differences. Holding the engine fixed
isolates what the benchmark is meant to show: the cost of Triton's HTTP frontend,
Python backend and request handling versus vLLM's own server. A TensorRT-LLM
backend is on the roadmap as a separate experiment.

## 3. Benchmark fairness rules

- Same model, same `max_model_len`, same GPU memory fraction for both engines.
- Same prompt text: vLLM applies the model's chat template, so the gateway
  renders the identical ChatML template for Triton (`inferscale/prompt.py`).
- Fixed output length (`ignore_eos`) so each request does the same decode work.
- Greedy decoding (`temperature: 0`).
- Every prompt is distinct, so prefix caching cannot skip prefill and inflate
  throughput.
- Warmup requests before each level; engines benchmarked one at a time.
- Results record the GPU, model, token counts and timestamp alongside the numbers.

## 4. Shed load at the gateway instead of queueing

**Decision.** Each gateway pod admits at most `maxInflight` requests and returns
`429` with `Retry-After` beyond that.

**Why.** Unbounded queueing inside the gateway turns overload into timeouts for
every client. A fast `429` keeps latency for admitted requests stable and gives
clients and the autoscaler an explicit signal. The engines keep their own
internal queues, which is where batching happens.

## 5. Surface upstream failures as real HTTP errors when streaming

**Decision.** For streaming requests the gateway waits for the engine's first
token before sending response headers.

**Why.** If the engine is down or rejects the prompt, the client gets a `503` or
`400` it can retry or fix, instead of a `200` stream that dies. Once tokens are
flowing, failures are reported in-band as an SSE `error` event followed by
`[DONE]`. Time to first token already includes this wait, so nothing is lost.

## 6. Stop sequences applied by the gateway for Triton

**Decision.** Triton's generate endpoint only accepts scalar parameters, so a
list of stop strings cannot be forwarded. The gateway applies them with a
streaming-safe filter that holds back any partial match across chunk boundaries
(`inferscale/stops.py`).

**Trade-off.** The engine keeps generating until the gateway closes the stream,
which can waste a few tokens of GPU work. Benchmarks do not use stop sequences.

## 7. Token accounting

vLLM reports exact usage. Triton's generate endpoint does not, so for streaming
the gateway counts streamed responses (the vLLM backend emits one token per
response without speculative decoding), and for non-streaming it estimates
(~4 characters per token) and marks the response `X-InferScale-Usage: estimated`.
The harness always streams, so throughput numbers use counted tokens on both paths.

## 8. Kubernetes specifics

- GPU deployments use the `Recreate` strategy: a rolling update would start a new
  pod that cannot get the GPU the old pod still holds.
- Startup probes allow 20 minutes for weight download and CUDA graph capture;
  readiness and liveness probes stay tight after startup.
- `/dev/shm` is enlarged with a memory-backed volume; the default 64 MB breaks
  multi-process engines.
- Gateway pods get a `preStop` sleep and a 60 s grace period so in-flight streams
  finish during rollouts.
- One uvicorn worker per gateway pod: scale with replicas so admission limits and
  metrics stay accurate per process.
