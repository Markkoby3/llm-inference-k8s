# Autoscaling

InferScale scales its two tiers on different signals because they saturate
differently.

| Tier | Signal | Why |
|---|---|---|
| Gateway | CPU utilization (HPA, 70%) | The gateway is a CPU-bound async proxy; CPU tracks its load directly. |
| GPU engines | Requests waiting in the engine queue (HPA on a Pods metric) | A saturated engine keeps GPU utilization pinned near 100% at any load, and its CPU barely moves. Queue depth is the first thing that rises when demand exceeds capacity. |

## Engine HPA

Enabled per engine in `values.yaml`:

```yaml
vllm:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 4
    metricName: vllm_num_requests_waiting
    targetAverageValue: "4"
    scaleDownStabilizationSeconds: 300
```

Scale-up is immediate (one pod per minute) because a new GPU pod needs minutes to
load weights. Scale-down waits five minutes so a short lull does not hand back a
GPU that will be needed again shortly.

## Exposing the metric to the HPA

The HPA reads metrics through the custom metrics API, which
[prometheus-adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
serves from Prometheus.

1. Install Prometheus (for example `kube-prometheus-stack`) and enable scraping:

   ```bash
   helm upgrade inferscale deploy/helm/inferscale --reuse-values \
     --set metrics.serviceMonitor.enabled=true
   ```

2. Install prometheus-adapter with rules that turn the engine metrics into
   per-pod custom metrics. vLLM exports names containing `:`, which the custom
   metrics API does not allow, so the rule renames them:

   ```yaml
   # prometheus-adapter values
   rules:
     custom:
       - seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
         resources:
           overrides:
             namespace: {resource: namespace}
             pod: {resource: pod}
         name:
           matches: "vllm:num_requests_waiting"
           as: "vllm_num_requests_waiting"
         metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
       - seriesQuery: 'nv_inference_pending_request_count{namespace!="",pod!=""}'
         resources:
           overrides:
             namespace: {resource: namespace}
             pod: {resource: pod}
         name:
           as: "nv_inference_pending_request_count"
         metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
   ```

3. Confirm the metric is visible:

   ```bash
   kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/vllm_num_requests_waiting"
   ```

## Choosing the target

`targetAverageValue: 4` means "add a replica when each engine pod has more than
four requests waiting on average." Calibrate it from a benchmark: find the
concurrency where p95 TTFT crosses your latency objective, then read the queue
depth at that point from `vllm:num_requests_waiting`. Set the target a little
below it so new capacity arrives before the objective is breached.
