#!/usr/bin/env bash
# Run by deploy.yml from the server directory after .env is regenerated and
# before production stops: a half-configured WeChat Pay must fail the deploy,
# not every payment after it. Disabled (the default) checks nothing.
set -euo pipefail

env_file="${1:-.env}"
env_value() { grep -E "^$1=" "$env_file" | cut -d= -f2- || true; }

if [ "$(env_value WECHATPAY_ENABLED)" != "true" ]; then
  exit 0
fi
for key in WECHATPAY_MCH_ID WECHATPAY_CERT_SERIAL_NO WECHATPAY_PUBLIC_KEY_ID WECHATPAY_API_V3_KEY WECHATPAY_NOTIFY_URL WECHAT_H5_APP_ID; do
  if [ -z "$(env_value "$key")" ]; then
    echo "$key is required when WECHATPAY_ENABLED=true"
    exit 1
  fi
done
for key in WECHATPAY_PRIVATE_KEY_PATH WECHATPAY_PUBLIC_KEY_PATH; do
  path_value="$(env_value "$key")"
  if [ -z "$path_value" ] || ! ${SUDO-sudo} test -r "$path_value"; then
    echo "$key must point to a readable PEM file on the VPS when WECHATPAY_ENABLED=true"
    exit 1
  fi
done
