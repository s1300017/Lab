#!/usr/bin/env bash
# 概要: Ollama 接続先を「ホスト直（localhost）」と「Docker コンテナ→ホスト（host.docker.internal）」で切り替えるユーティリティ
# 使い方:
#   ./scripts/switch_ollama_endpoint.sh docker [--container rag_backend] [--apply-env .env] [--restart-ollama] [--no-verify]
#   ./scripts/switch_ollama_endpoint.sh host   [--apply-env .env] [--no-verify]
#   ./scripts/switch_ollama_endpoint.sh verify [--container rag_backend]
#
# 注意:
# - 本スクリプトはデフォルトでは .env を変更しません（--apply-env <PATH> 指定時のみ書き換えます）。
# - brew サービスの再起動などの破壊的操作は --restart-ollama 指定時のみ行います。
# - バックエンドコードは `OLLAMA_BASE_URL` を参照します（backend/app/main.py の get_llm()）。

set -Eeuo pipefail

log() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err()  { printf "[ERROR] %s\n" "$*" 1>&2; }

usage() {
  cat <<'USAGE'
Ollama 接続先切り替えスクリプト

使い方:
  docker モード（コンテナ→ホストの接続先に切替）
    ./scripts/switch_ollama_endpoint.sh docker [--container rag_backend] [--apply-env .env] [--restart-ollama] [--no-verify]

  host モード（ホスト直で接続する場合の設定）
    ./scripts/switch_ollama_endpoint.sh host [--apply-env .env] [--no-verify]

  verify（疎通確認のみ）
    ./scripts/switch_ollama_endpoint.sh verify [--container rag_backend]

オプション:
  --apply-env <PATH>     指定した .env に OLLAMA_BASE_URL を追記/更新します（要注意: ファイルを書き換えます）
  --container <NAME>     疎通確認時に使用するコンテナ名（省略時は rag_backend を試行）
  --restart-ollama       Mac(brew) の Ollama サービスを 0.0.0.0:11434 で再起動します（launchctl/brew使用）
  --no-verify            切り替え後の疎通確認を省略
USAGE
}

SED_INPLACE() {
  # macOS(BSD sed) と GNU sed 両対応
  if sed --version >/dev/null 2>&1; then
    sed -i "${1}" "${2}"
  else
    sed -i '' "${1}" "${2}"
  fi
}

write_env_value() {
  # 引数: <env_file> <KEY> <VALUE>
  local env_file="$1" key="$2" value="$3"
  if [[ ! -f "$env_file" ]]; then
    warn "env ファイルが存在しないため新規作成します: $env_file"
    printf "%s=%s\n" "$key" "$value" > "$env_file"
    return 0
  fi
  if grep -q "^${key}=\|^# *${key}=" "$env_file"; then
    log "既存の ${key} を置換します: $env_file"
    SED_INPLACE "s|^#\? *${key}=.*|${key}=${value}|" "$env_file"
  else
    log "${key} を追記します: $env_file"
    printf "%s=%s\n" "$key" "$value" >> "$env_file"
  fi
}

verify_host() {
  # ホスト直の疎通確認
  log "ホストで Ollama バージョン確認: http://localhost:11434/api/version"
  if ! curl -sSf http://localhost:11434/api/version >/dev/null; then
    err "ホストの Ollama に接続できません。ollama serve の状態を確認してください。"
    return 1
  fi
  log "OK: ホストの Ollama 応答あり"
}

verify_container() {
  # コンテナ→ホストの疎通確認
  local container_name="$1"
  if [[ -z "$container_name" ]]; then
    container_name="rag_backend"
  fi
  log "コンテナ ${container_name} からホストの Ollama を確認: http://host.docker.internal:11434/api/version"
  if ! docker exec -it "$container_name" curl -sSf http://host.docker.internal:11434/api/version >/dev/null; then
    err "コンテナ ${container_name} から Ollama に接続できません。コンテナ名/ネットワーク/公開設定を確認してください。"
    return 1
  fi
  log "OK: コンテナ ${container_name} からの疎通成功"
}

restart_ollama_service() {
  # Mac(brew) 環境での公開バインド（0.0.0.0:11434）恒久化
  if ! command -v brew >/dev/null 2>&1; then
    warn "brew が見つかりません。手動起動中の ollama を使用している可能性があります。"
  fi
  log "launchd に OLLAMA_HOST=0.0.0.0:11434 を設定"
  launchctl setenv OLLAMA_HOST 0.0.0.0:11434 || true
  log "brew サービスの ollama を再起動"
  brew services restart ollama || true
  sleep 2
  log "LISTEN 状態を確認"
  lsof -nP -iTCP:11434 -sTCP:LISTEN || true
}

main() {
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi
  local mode="$1"; shift || true
  local env_path="" container_name="" do_verify=1 do_restart_ollama=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --apply-env)
        env_path="$2"; shift 2;;
      --container)
        container_name="$2"; shift 2;;
      --restart-ollama)
        do_restart_ollama=1; shift;;
      --no-verify)
        do_verify=0; shift;;
      -h|--help)
        usage; exit 0;;
      *)
        err "不明な引数: $1"; usage; exit 1;;
    esac
  done

  case "$mode" in
    docker)
      # コンテナ→ホスト接続向け
      local value="http://host.docker.internal:11434"
      log "切替: OLLAMA_BASE_URL=${value} (Docker コンテナ→ホスト)"
      if [[ -n "$env_path" ]]; then
        write_env_value "$env_path" OLLAMA_BASE_URL "$value"
      else
        log "環境変数適用例（シェル一時適用）: export OLLAMA_BASE_URL=${value}"
      fi
      if [[ "$do_restart_ollama" -eq 1 ]]; then
        restart_ollama_service
      fi
      if [[ "$do_verify" -eq 1 ]]; then
        verify_container "$container_name" || exit 1
      fi
      ;;
    host)
      # ホスト直接続向け
      local value="http://localhost:11434"
      log "切替: OLLAMA_BASE_URL=${value} (ホスト直)"
      if [[ -n "$env_path" ]]; then
        write_env_value "$env_path" OLLAMA_BASE_URL "$value"
      else
        log "環境変数適用例（シェル一時適用）: export OLLAMA_BASE_URL=${value}"
      fi
      if [[ "$do_verify" -eq 1 ]]; then
        verify_host || exit 1
      fi
      ;;
    verify)
      # 疎通確認のみ
      verify_host || true
      verify_container "$container_name" || true
      ;;
    *)
      err "不明なモード: $mode"; usage; exit 1;;
  esac

  log "完了"
}

main "$@"
