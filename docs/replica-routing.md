# Replica routing

When an engine runs more than one replica, the gateway routes each request to a
specific replica itself instead of leaving it to the Kubernetes Service. The
reason is the engines' prefix (KV) cache: vLLM and TensorRT-LLM skip prefill for a
prompt prefix they have already processed. A ClusterIP Service spreads
connections at random, so two requests with the same long system prompt usually
land on different pods and both pay full prefill.

## Policies

Set with `INFERSCALE_ROUTING_POLICY` (Helm: `routing.policy`).

| Policy | How it picks a replica | Use when |
|---|---|---|
| `prefix_affinity` (default) | Rendezvous hash of the conversation prefix, with a load bound | Requests share system prompts or few-shot context (agents, RAG, chat apps) |
| `least_inflight` | Fewest requests in flight | Prompts share little; balance matters most |
| `round_robin` | Next replica in turn | Baseline |

**The prefix key** is everything before the newest message (system prompt,
instructions, earlier turns), cut to `routing.prefixChars` characters. Two
requests with the same key share the token sequence an engine can cache.

**Rendezvous hashing** ranks replicas by `hash(key, replica)` and takes the top
one. When the HPA adds or removes a replica, only the keys that replica owned
move; every other replica keeps its warm cache. A modulo hash would reshuffle
almost every key on each scale event.

**The load bound.** Pure affinity sends every request for a popular prefix to
one replica. Each replica may therefore take a request only while its in-flight
count is below `loadFactor` × the pool average (consistent hashing with bounded
loads, Mirrokni, Thorup and Zadimoghaddam, 2018). Otherwise the request spills to
the next replica in that key's ranking, so even spillover lands on a stable,
increasingly warm second choice.

## Measured in a simulation

`inferscale-bench routing` runs the gateway's real `ReplicaPool` over mock
replicas that model a prefix cache (an LRU of prefixes per replica; a miss pays
prefill for the whole prompt, a hit only for the new message) and finite batch
slots (excess load queues). The workload is agent-style: each request uses one of
64 system prompts of ~3,000 characters, chosen with Zipf popularity, plus a short
unique question.

This isolates the routing decision. It is a model, not a GPU measurement: real
gains depend on the engine's KV-cache capacity, block size and eviction.

**Typical skew (Zipf s = 1.1):**

| Policy | Cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | Req/s | Load imbalance |
|---|---:|---:|---:|---:|---:|
| round_robin | 48% | 86.9 | 104.9 | 501.3 | 1.0x |
| least_inflight | 46% | 86.9 | 88.9 | 525.5 | 1.04x |
| prefix_affinity | 71% | 11.9 | 89.4 | 791.3 | 1.37x |
| prefix_affinity, unbounded | 81% | 14.8 | 94.2 | 844.1 | 1.56x |

**One dominant prefix (Zipf s = 2.0):**

| Policy | Cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | Req/s | Load imbalance |
|---|---:|---:|---:|---:|---:|
| round_robin | 89% | 11.8 | 88.4 | 1286.8 | 1.0x |
| least_inflight | 89% | 11.7 | 88.0 | 1339.4 | 1.08x |
| prefix_affinity | 94% | 11.8 | 87.3 | 1595.6 | 1.47x |
| prefix_affinity, unbounded | 97% | 31.1 | 45.7 | 975.9 | 2.65x |

What the numbers say:

- With typical skew, bounded prefix affinity raises the cache hit rate from 48%
  to 71%, cuts median time to first token from 87 ms to 12 ms, and serves 58% more
  requests per second than round robin.
- The bound is what makes affinity safe. With one dominant prefix, unbounded
  affinity piles 2.65x the average load onto one replica and throughput falls 39%
  below the bounded version, despite a higher hit rate.
- p95 latency barely moves under typical skew because misses still pay full
  prefill: a fifth to a third of requests miss under any policy. A bigger cache
  per replica moves the tail, not the router.

Reproduce:

```bash
inferscale-bench routing                 # typical skew
inferscale-bench routing --zipf 2.0      # one dominant prefix
inferscale-bench routing --cache 16 --replicas 8
```

## Replica discovery on Kubernetes

With more than one replica or autoscaling enabled, the chart points the gateway
at `dns://<release>-<engine>-headless:8000`. A headless Service returns one DNS
record per **ready** pod, and the gateway re-resolves it every
`INFERSCALE_DISCOVERY_REFRESH_S` (10 s by default):

- **Scale-up:** a new pod joins once its readiness probe passes.
- **Scale-down:** a removed pod drains. It keeps its in-flight streams, takes no
  new requests, and its connections close once it is idle.
- **Failure:** a replica that refuses connections leaves rotation for a cooldown,
  and the request retries once on another replica. Timeouts and engine errors are
  not retried, because the engine may already have spent GPU time on them.

Outside Kubernetes, list replica URLs directly:
`INFERSCALE_VLLM_URL=http://gpu-a:8000,http://gpu-b:8000`.

## Observability

| Metric | Meaning |
|---|---|
| `inferscale_routing_decisions_total{backend,policy,decision}` | `affinity` vs `spill`; a rising spill share means a hot prefix or an undersized pool |
| `inferscale_replica_requests_total{backend,replica}` | Load per replica |
| `inferscale_replica_inflight_requests{backend,replica}` | Current load per replica |
| `inferscale_replica_failovers_total{backend}` | Replicas taken out of rotation |

`/readyz` lists every replica with its readiness, load and rotation state.
