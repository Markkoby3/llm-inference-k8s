{{- define "inferscale.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "inferscale.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "inferscale.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "inferscale.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: inferscale
{{- end -}}

{{/* Selector labels for one component: pass (dict "ctx" $ "component" "gateway"). */}}
{{- define "inferscale.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inferscale.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* Comma-separated backend list the gateway is configured with. */}}
{{- define "inferscale.backends" -}}
{{- if and .Values.vllm.enabled .Values.triton.enabled -}}
vllm,triton
{{- else if .Values.vllm.enabled -}}
vllm
{{- else if .Values.triton.enabled -}}
triton
{{- else -}}
mock
{{- end -}}
{{- end -}}

{{- define "inferscale.defaultBackend" -}}
{{- $backends := include "inferscale.backends" . -}}
{{- if not (has .Values.gateway.defaultBackend (splitList "," $backends)) -}}
{{- fail (printf "gateway.defaultBackend %q is not an enabled backend (enabled: %s)" .Values.gateway.defaultBackend $backends) -}}
{{- end -}}
{{- .Values.gateway.defaultBackend -}}
{{- end -}}

{{/* Volumes shared by the GPU engines: model cache and enlarged /dev/shm. */}}
{{- define "inferscale.engineVolumes" -}}
- name: model-cache
{{- if .Values.model.cache.persistentVolumeClaim }}
  persistentVolumeClaim:
    claimName: {{ .Values.model.cache.persistentVolumeClaim }}
{{- else }}
  emptyDir:
    sizeLimit: {{ .Values.model.cache.sizeLimit }}
{{- end }}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.gpu.shmSize }}
{{- end -}}

{{- define "inferscale.engineVolumeMounts" -}}
- name: model-cache
  mountPath: /root/.cache/huggingface
- name: dshm
  mountPath: /dev/shm
{{- end -}}

{{- define "inferscale.engineEnv" -}}
- name: HF_HOME
  value: /root/.cache/huggingface
{{- if .Values.model.hfTokenSecret }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.model.hfTokenSecret }}
      key: HF_TOKEN
{{- end }}
{{- end -}}

{{/*
HPA for a GPU engine, scaling on a per-pod queue-depth metric served by
prometheus-adapter. Pass (dict "ctx" $ "component" "vllm" "autoscaling" .Values.vllm.autoscaling).
*/}}
{{- define "inferscale.engineHPA" -}}
{{- $fullname := include "inferscale.fullname" .ctx -}}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ $fullname }}-{{ .component }}
  labels:
    {{- include "inferscale.labels" .ctx | nindent 4 }}
    app.kubernetes.io/component: {{ .component }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ $fullname }}-{{ .component }}
  minReplicas: {{ .autoscaling.minReplicas }}
  maxReplicas: {{ .autoscaling.maxReplicas }}
  metrics:
    - type: Pods
      pods:
        metric:
          name: {{ .autoscaling.metricName }}
        target:
          type: AverageValue
          averageValue: {{ .autoscaling.targetAverageValue | quote }}
  behavior:
    # GPU pods take minutes to become ready: scale up immediately, but wait
    # before giving a GPU back so a brief lull does not cause flapping.
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: {{ .autoscaling.scaleDownStabilizationSeconds }}
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
{{- end -}}

{{- define "inferscale.engineScheduling" -}}
{{- with .Values.gpu.runtimeClassName }}
runtimeClassName: {{ . }}
{{- end }}
{{- with .Values.gpu.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.gpu.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
