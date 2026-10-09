# GPU benchmark runbook

How to produce the vLLM vs. Triton numbers on a single rented GPU, from a clean
machine to a committed results table. Budget: about 1–2 hours of GPU time.

Two paths:

| Path | What you get | When to use |
|---|---|---|
| **A. Docker Compose** | Numbers in ~20 minutes | First run, sanity check |
| **B. k3s + Helm** | The same numbers, served from Kubernetes | The real deployment story |

## Machine

Any Linux VM with one NVIDIA GPU (≥ 16 GB), root access, the NVIDIA driver,
Docker and the NVIDIA Container Toolkit. Lambda Cloud, or an AWS `g6.xlarge`
(L4 24 GB) / GCP `g2-standard-8` (L4) with a deep-learning image, all work.
Container-only "pods" (no Docker inside) do not work for these steps.

Check the GPU and toolkit before anything else:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Record the GPU for the results:

```bash
export GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
```

## Path A: Docker Compose

```bash
git clone https://github.com/Markkoby3/llm-inference-k8s && cd llm-inference-k8s
docker compose -f deploy/compose/docker-compose.gpu.yaml up -d --build
docker compose -f deploy/compose/docker-compose.gpu.yaml ps   # wait until all are healthy
```

First start downloads ~3 GB of weights and warms up both engines (5–10 min).

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
curl -s localhost:8080/readyz   # {"ready":true,"backends":{"vllm":true,"triton":true}}
```

Run each backend separately, never both at once (they share the GPU):

```bash
inferscale-bench run --backend vllm   --gpu "$GPU" --concurrency 1,4,16,64 --requests 200
inferscale-bench run --backend triton --gpu "$GPU" --concurrency 1,4,16,64 --requests 200
inferscale-bench compare benchmarks/results/*.json --out benchmarks/results/comparison.md
```

Optional: measure the gateway's own overhead by pointing the harness straight at
vLLM's server and comparing with the `vllm` run above:

```bash
inferscale-bench run --url http://localhost:8000 --label vllm-direct --gpu "$GPU" \
  --concurrency 1,4,16,64 --requests 200
```

## Path B: k3s + Helm

### 1. Kubernetes with GPU support

Install the NVIDIA Container Toolkit **before** k3s, so k3s detects the NVIDIA runtime:

```bash
curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml && sudo chmod 644 $KUBECONFIG
kubectl get runtimeclass nvidia || kubectl apply -f - <<'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata: {name: nvidia}
handler: nvidia
EOF
```

### 2. Device plugin with time-slicing

One GPU must be schedulable by two pods (vLLM and Triton):

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  --set runtimeClassName=nvidia \
  --set-file config.map.config=deploy/gpu/time-slicing.yaml \
  --set config.default=config
kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # expect 2
```

### 3. Gateway image

```bash
docker build -t inferscale-gateway:dev .
docker save inferscale-gateway:dev | sudo k3s ctr images import -
```

### 4. Deploy

```bash
helm upgrade --install inferscale deploy/helm/inferscale \
  -f deploy/helm/inferscale/values-single-gpu.yaml \
  --set gateway.image.repository=inferscale-gateway \
  --set gateway.image.tag=dev --set gateway.image.pullPolicy=Never
kubectl get pods -w   # engines take several minutes on first start
```

### 5. Benchmark through the cluster

```bash
kubectl port-forward svc/inferscale-gateway 8080:80 &
inferscale-bench run --backend vllm   --label k8s-vllm   --gpu "$GPU"
inferscale-bench run --backend triton --label k8s-triton --gpu "$GPU"
inferscale-bench compare benchmarks/results/k8s-*.json
```

## Publish the results

`benchmarks/results/` is git-ignored so scratch runs never land in history.
Copy the runs you stand behind into `benchmarks/published/`, paste the comparison
table into the README's Results section with the GPU and date, and commit:

```bash
mkdir -p benchmarks/published && cp benchmarks/results/*.{json,md} benchmarks/published/
nvidia-smi > benchmarks/published/nvidia-smi.txt
```

Then **shut the instance down**.

## Sanity checks before trusting a run

- `errors` is 0/N at every level. Errors mean the gateway shed load (429) or an
  engine timed out; the throughput for that level is not comparable.
- `mean_output_tokens` equals `--max-tokens` (256 by default). If not, `ignore_eos`
  was not honored and backends did different amounts of work.
- TTFT at concurrency 1 should be tens of milliseconds for a 1.5B model. Seconds
  means the engine was still warming up: increase `--warmup`.
- Run each configuration twice. If throughput differs by more than ~5%, something
  else is using the GPU.
