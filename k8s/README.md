# Deploying

```
k8s/
  base/                 namespace, ConfigMap, Redis, backend (2 replicas), frontend — same everywhere
  overlays/gke/         Artifact Registry images tagged by git SHA, L7 Ingress, NEGs, deploy.sh
  overlays/kind/        local images + nginx Ingress on bible.local
  secret.example.yaml   copy to k8s/secret.yaml (gitignored) and fill in
```

Secrets are applied separately from kustomize so the file holding real keys never has to exist in the tree for `kubectl kustomize` to work.

## GKE (Autopilot)

Autopilot bills per pod (no node management, no idle nodes). With the manifests as
written the cluster runs ~1.25 vCPU and ~1.5 GiB in total — roughly $1.50/day plus
about $0.60/day for the load balancer. The first cluster in a project is covered by
GKE's free tier credit. **Delete the cluster when you are done** (last section).

The app is exposed through Google's external Application Load Balancer (L7):
`/api/*` goes straight to the backend, everything else to the frontend, so the browser
and the API share one origin. The LB adds `X-Forwarded-For`, which is what the backend's
per-IP rate limit keys on (`CLIENT_IP_XFF_INDEX=-2`, set in `overlays/gke/configmap-patch.yaml`,
because Google appends `<client>, <lb-ip>`).

### 0. Prerequisites

- A GCP project with billing enabled. Use a personal project for a portfolio deploy.
- `gcloud`, `kubectl` and the GKE auth plugin:

```bash
gcloud components install gke-gcloud-auth-plugin
```

### 1. One-time project setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=$(gcloud config get-value project) REGION=us-east1
```

```bash
gcloud services enable container.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

```bash
gcloud artifacts repositories create bible --repository-format=docker --location=$REGION
```

Create the cluster (5–8 minutes). `us-east1` keeps the hop to the Pinecone index in AWS us-east-1 short.

```bash
gcloud container clusters create-auto bible --region=$REGION
```

```bash
gcloud container clusters get-credentials bible --region=$REGION
```

`kubectl config current-context` should now show `gke_<project>_us-east1_bible`.

### 2. Secrets

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
```

Fill in `OPENAI_API_KEY` and `PINECONE_API_KEY`, then:

```bash
kubectl apply -f k8s/base/namespace.yaml && kubectl apply -f k8s/secret.yaml
```

### 3. Build and deploy

```bash
k8s/overlays/gke/deploy.sh
```

The script builds both images with Cloud Build (x86, ~3–5 min, in parallel), tags them
with the git SHA, renders `overlays/gke` with your project/region, applies it, waits for
the rollouts and then for the load balancer to answer, and prints the public URL. The
first LB provisioning takes 5–10 minutes; later deploys only wait for new pods. Re-run
after every commit; `SKIP_BUILD=1 TAG=<sha> k8s/overlays/gke/deploy.sh` redeploys an
image that already exists (that is also how you roll back).

### 4. Verify

```bash
kubectl -n bible-reference-ai get pods,svc,ingress
```

`http://<ip>/` is the UI; `http://<ip>/api/cache/stats` shows hit counts and the
running config. The per-IP limit is 20 requests/min on `POST /api/query` (429 +
`Retry-After` beyond that) and 60/min across all clients, both tunable in the ConfigMap.

To run the cache benchmark against the cluster, port-forward the backend Service. Port-forwarded
requests all look like `127.0.0.1` to the pod, so lift the per-IP limit for the benchmark window
and put it back afterwards (each `set env` triggers a rolling restart):

```bash
kubectl -n bible-reference-ai set env deploy/bible-backend RATE_LIMIT_PER_MINUTE=0 RATE_LIMIT_GLOBAL_PER_MINUTE=0
```

```bash
kubectl -n bible-reference-ai port-forward svc/bible-backend-svc 8001:8000
```

```bash
cd backend && uv run python -m scripts.benchmark_cache --base-url http://127.0.0.1:8001 \
    --flush --flush-cmd "kubectl -n bible-reference-ai exec deploy/bible-redis -- redis-cli FLUSHALL" \
    --out eval_results/cache_benchmark_gke.json --mixed 300 --concurrency 4
```

```bash
kubectl -n bible-reference-ai set env deploy/bible-backend RATE_LIMIT_PER_MINUTE- RATE_LIMIT_GLOBAL_PER_MINUTE-
```

`--flush-cmd` empties the Redis *inside the cluster* (the script's own flush only reaches a local Redis), and `--out` keeps the cluster results separate from the local ones. A port-forward pins every request to one backend pod, so this measures the production path (Redis, Pinecone and OpenAI from inside GKE) but not load spreading across the two replicas.

Logs:

```bash
kubectl -n bible-reference-ai logs deploy/bible-backend -f
```

### 5. Tear down

```bash
gcloud container clusters delete bible --region=$REGION --quiet
```

Deleting the cluster also removes the load balancer it created. Images in Artifact
Registry cost cents a month; remove the repo if you want nothing left:

```bash
gcloud artifacts repositories delete bible --location=$REGION --quiet
```

### Notes

- Autopilot raises pod requests below its floor (250m CPU / 512Mi) automatically and sets limits equal to requests; the warnings it prints for Redis are expected.
- Services carry `cloud.google.com/neg: '{"ingress": true}'` so the LB targets pod IPs directly (container-native load balancing). The LB's health checks are derived from the readiness probes (`/health` on the backend, `/` on the frontend).
- Image tags are immutable git SHAs, so a rollout is always traceable to a commit.
- Both Dockerfiles ship with a `.gcloudignore`; without it Cloud Build would upload `.venv` / `node_modules`.

## kind (local)

```bash
kind create cluster --config kind-config.yaml
docker build -t bible-backend:dev backend && docker build -t bible-frontend:dev frontend
kind load docker-image bible-backend:dev bible-frontend:dev --name bible-reference-ai
kubectl apply -k k8s/overlays/kind && kubectl apply -f k8s/secret.yaml
```

Add `127.0.0.1 bible.local` to `/etc/hosts` and install ingress-nginx for kind; the Ingress routes `/` to the frontend and `/api` to the backend.
