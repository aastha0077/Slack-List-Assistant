#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
PACKAGE_FILE="${PROJECT_DIR}/slack-list-assistant.zip"

FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-slack-list-assistant}"
RUNTIME="${LAMBDA_RUNTIME:-python3.12}"
TIMEOUT_SECONDS="${LAMBDA_TIMEOUT_SECONDS:-60}"

DEPLOY_BUCKET="${LAMBDA_DEPLOY_BUCKET:-slack-list-assistant-deploy-529088259575}"
S3_KEY="slack-list-assistant/$(date +%Y%m%d-%H%M%S)-slack-list-assistant.zip"

echo "Cleaning previous Lambda build..."
rm -rf "${BUILD_DIR}" "${PACKAGE_FILE}"
mkdir -p "${BUILD_DIR}"

echo "Installing Lambda Python 3.12 Linux x86_64 dependencies..."

python -m pip install \
  --requirement "${PROJECT_DIR}/requirements.txt" \
  --target "${BUILD_DIR}" \
  --ignore-installed \
  --no-warn-conflicts \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all:

echo "Copying application files..."

cp "${PROJECT_DIR}/list_schema.txt" "${BUILD_DIR}/"
cp -R "${PROJECT_DIR}/src" "${BUILD_DIR}/"
cp "${PROJECT_DIR}/src/handler.py" "${BUILD_DIR}/handler.py"

find "${BUILD_DIR}" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "${BUILD_DIR}" -type f -name '*.pyc' -delete

echo "Creating Lambda package..."

(
  cd "${BUILD_DIR}"
  zip -qr "${PACKAGE_FILE}" .
)

echo "Package size:"
ls -lh "${PACKAGE_FILE}"

echo "Validating Lambda package..."

if unzip -l "${PACKAGE_FILE}" | grep -E 'darwin|cpython-313' >/dev/null; then
  echo "ERROR: macOS/Python 3.13 binary detected in Lambda package." >&2
  exit 1
fi

# PNG visualization is a required feature. Never publish a package in which
# the Python module, its native extension, or its Lambda-compatible wheel is
# absent. Keep the lazy runtime import; enforce the dependency at build time.
MATPLOTLIB_WHEEL="$(unzip -Z1 "${PACKAGE_FILE}" | grep -E '^matplotlib-[^/]+\.dist-info/WHEEL$' | sed -n '1p' || true)"
if [[ -z "${MATPLOTLIB_WHEEL}" ]] ||
   ! unzip -Z1 "${PACKAGE_FILE}" | grep -E '^matplotlib/__init__\.py$' >/dev/null ||
   ! unzip -Z1 "${PACKAGE_FILE}" | grep -E '^matplotlib/_path\.cpython-312-x86_64-linux-gnu\.so$' >/dev/null ||
   ! unzip -p "${PACKAGE_FILE}" "${MATPLOTLIB_WHEEL}" |
       grep -E '^Tag: cp312-cp312-manylinux(2014|_2_17)_x86_64$' >/dev/null; then
  echo "ERROR: Lambda package is missing Python 3.12 x86_64 Matplotlib." >&2
  exit 1
fi

if [[ "${LAMBDA_BUILD_ONLY:-0}" == "1" ]]; then
  echo "Lambda package validated; build-only mode, no AWS upload or deployment."
  exit 0
fi

echo "Uploading package to S3..."

aws s3 cp \
  "${PACKAGE_FILE}" \
  "s3://${DEPLOY_BUCKET}/${S3_KEY}"

echo "Deploying ${FUNCTION_NAME} from S3..."

if aws lambda get-function --function-name "${FUNCTION_NAME}" >/dev/null 2>&1; then

  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --s3-bucket "${DEPLOY_BUCKET}" \
    --s3-key "${S3_KEY}"

  aws lambda wait function-updated \
    --function-name "${FUNCTION_NAME}"

  aws lambda update-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --runtime "${RUNTIME}" \
    --handler "handler.lambda_handler" \
    --timeout "${TIMEOUT_SECONDS}"

  aws lambda wait function-updated \
    --function-name "${FUNCTION_NAME}"

else

  echo "ERROR: Lambda function ${FUNCTION_NAME} does not exist." >&2
  exit 1

fi

DEPLOYED_HANDLER="$(
  aws lambda get-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --query Handler \
    --output text
)"

DEPLOYED_RUNTIME="$(
  aws lambda get-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --query Runtime \
    --output text
)"

DEPLOYED_ARCHITECTURE="$(
  aws lambda get-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --query 'Architectures[0]' \
    --output text
)"

DEPLOYED_TIMEOUT="$(
  aws lambda get-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --query Timeout \
    --output text
)"

if [[ "${DEPLOYED_HANDLER}" != "handler.lambda_handler" ||
      "${DEPLOYED_RUNTIME}" != "python3.12" ||
      "${DEPLOYED_ARCHITECTURE}" != "x86_64" ||
      "${DEPLOYED_TIMEOUT}" != "${TIMEOUT_SECONDS}" ]]; then

  echo "ERROR: Lambda configuration verification failed." >&2
  echo "Handler: ${DEPLOYED_HANDLER}" >&2
  echo "Runtime: ${DEPLOYED_RUNTIME}" >&2
  echo "Architecture: ${DEPLOYED_ARCHITECTURE}" >&2
  echo "Timeout: ${DEPLOYED_TIMEOUT}" >&2
  exit 1
fi

echo ""
echo "Deployment successful."
echo "Function: ${FUNCTION_NAME}"
echo "Runtime: ${DEPLOYED_RUNTIME}"
echo "Architecture: ${DEPLOYED_ARCHITECTURE}"
echo "Handler: ${DEPLOYED_HANDLER}"
echo "Timeout: ${DEPLOYED_TIMEOUT}s"
echo "S3 package: s3://${DEPLOY_BUCKET}/${S3_KEY}"
