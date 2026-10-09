CHART := deploy/helm/inferscale
IMAGE ?= inferscale-gateway:dev
URL ?= http://localhost:8080
BACKEND ?=
CONCURRENCY ?= 1,4,16,64
REQUESTS ?= 200
GPU ?=

.PHONY: install lint test check run-mock bench bench-compare docker-build \
        kind-up kind-deploy kind-down helm-lint

install:  ## Install the package with dev tools
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

check: lint test helm-lint

run-mock:  ## Run the gateway locally with the mock backend
	INFERSCALE_BACKENDS=mock python -m inferscale

bench:  ## Benchmark a running gateway: make bench BACKEND=vllm GPU="L4 24GB"
	inferscale-bench run --url $(URL) $(if $(BACKEND),--backend $(BACKEND)) \
	  --concurrency $(CONCURRENCY) --requests $(REQUESTS) $(if $(GPU),--gpu "$(GPU)")

bench-compare:  ## Side-by-side table of every saved run
	inferscale-bench compare benchmarks/results/*.json --out benchmarks/results/comparison.md

docker-build:
	docker build -t $(IMAGE) .

helm-lint:
	helm lint --strict $(CHART)
	helm lint --strict $(CHART) -f $(CHART)/values-mock.yaml
	helm lint --strict $(CHART) -f $(CHART)/values-single-gpu.yaml

kind-up:
	kind create cluster --name inferscale --config deploy/kind/kind-config.yaml

kind-deploy: docker-build  ## Deploy the mock profile to kind
	kind load docker-image $(IMAGE) --name inferscale
	helm upgrade --install inferscale $(CHART) -f $(CHART)/values-mock.yaml --wait --timeout 3m

kind-down:
	kind delete cluster --name inferscale
