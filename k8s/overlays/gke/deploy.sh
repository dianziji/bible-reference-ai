#!/usr/bin/env bash
# Build both images with Cloud Build, tag them with the current git SHA, and
# roll them out to the GKE cluster kubectl is pointed at.
#
# One-time setup (APIs, Artifact Registry repo, cluster, secret) is in k8s/README.md.
#
#   PROJECT_ID=my-project REGION=us-east1 k8s/overlays/gke/deploy.sh
#
# PROJECT_ID defaults to the active gcloud project, REGION to us-east1, TAG to
# the short git SHA. SKIP_BUILD=1 TAG=<sha> redeploys an image that already exists.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east1}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
NAMESPACE="bible-reference-ai"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/bible"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is not set and gcloud has no active project" >&2
  exit 1
fi

echo "▶ project=${PROJECT_ID} region=${REGION} tag=${TAG}"

# Cloud Build compiles on x86 machines, so the images run on GKE nodes
# regardless of the arm64 Mac they were submitted from. Both builds run at once.
if [[ -z "${SKIP_BUILD:-}" ]]; then
  # In a brand-new project two simultaneous submits race to create the staging
  # bucket and one of them fails with "bucket does not exist"; create it first.
  gcloud storage buckets create "gs://${PROJECT_ID}_cloudbuild" --project "${PROJECT_ID}" >/dev/null 2>&1 || true
  gcloud builds submit "${ROOT}/backend"  --tag "${IMAGE_BASE}/backend:${TAG}"  --quiet & backend_build=$!
  gcloud builds submit "${ROOT}/frontend" --tag "${IMAGE_BASE}/frontend:${TAG}" --quiet & frontend_build=$!
  wait "${backend_build}"
  wait "${frontend_build}"
fi

echo "▶ applying manifests"
kubectl kustomize "${ROOT}/k8s/overlays/gke" \
  | sed -e "s#__REGION__#${REGION}#g" -e "s#__PROJECT_ID__#${PROJECT_ID}#g" -e "s#__IMAGE_TAG__#${TAG}#g" \
  | kubectl apply -f -

if ! kubectl -n "${NAMESPACE}" get secret bible-secrets >/dev/null 2>&1; then
  echo "✗ secret bible-secrets is missing — pods will not start." >&2
  echo "  cp k8s/secret.example.yaml k8s/secret.yaml, fill it in, then: kubectl apply -f k8s/secret.yaml" >&2
  exit 1
fi

kubectl -n "${NAMESPACE}" rollout status deploy/bible-redis    --timeout=5m
kubectl -n "${NAMESPACE}" rollout status deploy/bible-backend  --timeout=5m
kubectl -n "${NAMESPACE}" rollout status deploy/bible-frontend --timeout=5m

# The first time, Google needs 5–10 minutes to provision the load balancer and
# for its health checks to pass; later deploys only wait for the new pods.
echo "▶ waiting for the load balancer"
ip=""
for _ in $(seq 1 120); do
  ip="$(kubectl -n "${NAMESPACE}" get ingress bible-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
  if [[ -n "${ip}" ]] \
     && curl -fsS -m 5 "http://${ip}/" >/dev/null 2>&1 \
     && curl -fsS -m 5 "http://${ip}/api/cache/stats" >/dev/null 2>&1; then
    echo "✅ http://${ip}/"
    exit 0
  fi
  sleep 5
done

echo "⚠ the load balancer is still warming up (ip=${ip:-pending}); check:"
echo "  kubectl -n ${NAMESPACE} get ingress bible-ingress"
echo "  kubectl -n ${NAMESPACE} describe ingress bible-ingress | grep -A3 backends"
