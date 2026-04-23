# llm-dashboard

A single-binary, passive monitoring dashboard for locally-hosted
vLLM / OpenAI-compatible LLM servers on DGX Spark clusters.

It sits in front of your LLM as a reverse proxy and records every request's
tokens/second, prompt/completion token counts, and prefill/decode timing.
In parallel it SSHs into each configured Spark and samples `nvidia-smi`
GPU temperature so each row shows the peak GPU temp seen during that request.

- **No instrumentation needed in your LLM server** — just re-point clients at the proxy.
- **Single Go binary, pure-Go SQLite** — no CGO, drop it anywhere.
- **Table UI** with live averages footer at `http://localhost:7000`.

---

## Build

```bash
go build -o llm-dashboard .
```

That's it. One static binary.

## Run

```bash
./llm-dashboard
# dashboard  -> http://localhost:7000
# llm proxy -> http://localhost:7001
```

Flags:

- `-ui :7000`     dashboard HTTP listener
- `-proxy :7001`  the reverse-proxy your clients should talk to
- `-db ./llm-dashboard.db` SQLite path

## First-time setup

1. Open `http://localhost:7000` in a browser.
2. Click **Settings**.
3. Set **Upstream LLM URL** to where your vLLM (or OpenAI-compatible) server is actually listening, e.g. `http://localhost:8000`.
4. Add each DGX Spark node (name, host, user, port, GPU index).
5. Save.

## Use

Point any OpenAI-compatible client at the **proxy** instead of the upstream:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:7001/v1", api_key="anything")
```

or

```bash
curl http://localhost:7001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

Every request flows through transparently and shows up in the dashboard within ~2s.

## What's recorded

| column | source |
| --- | --- |
| task | first ~120 chars of the last user message / prompt |
| model | request's `model` field or response's `model` |
| t/s | `completion_tokens / decode_time` (falls back to wall-clock) |
| prompt / out | `usage.prompt_tokens`, `usage.completion_tokens` |
| prefill | upstream send → first SSE chunk (streaming only) |
| total | full wall-clock request time |
| °C sparkN | **peak** `nvidia-smi` temp sampled every 500ms on that Spark during the request window |

Footer row shows **average t/s** and **average peak temp per Spark** across all records currently rendered.

Heat coloring: <70°C green, 70–85°C amber, ≥85°C red.

## SSH setup for GPU temps

The dashboard's host machine needs passwordless SSH into every Spark. On the dashboard host:

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519   # if you don't have one
ssh-copy-id ubuntu@spark1
ssh-copy-id ubuntu@spark2
# verify:
ssh ubuntu@spark1 'nvidia-smi --query-gpu=temperature.gpu -i 0 --format=csv,noheader,nounits'
```

Notes:

- Host keys are **not** verified (`ssh.InsecureIgnoreHostKey`). That's fine on a trusted LAN; if you'd rather pin known_hosts, edit `ssh_probe.go`.
- Default key paths tried: `~/.ssh/id_ed25519`, `~/.ssh/id_rsa`. Override via the `key_path` field if you keep a dedicated key.
- Password auth is not supported by design — set up keys.

## Where data lives

`./llm-dashboard.db` (SQLite) contains three tables: `records`, `sparks`, `config`. Back it up, grep it, import it to pandas — it's just SQLite.

```bash
sqlite3 llm-dashboard.db 'SELECT ts, model, tokens_per_second, gpu_temps FROM records ORDER BY id DESC LIMIT 20'
```

## Troubleshooting

- **"no upstream_url configured"** → open Settings, set it, save.
- **GPU temp columns show `—`** → your SSH key isn't reaching the Spark, or `nvidia-smi` isn't on PATH for that SSH user. Test manually with the command above.
- **t/s shows `—` for a row** → the upstream didn't return `usage` (some servers omit it on streaming). Ask vLLM for usage by adding `"stream_options": {"include_usage": true}` to your requests, or the dashboard will fall back to counting streamed chunks.
- **Streaming response hangs** → disable compression on the upstream (the proxy already sets `Accept-Encoding:` to empty).

## Cross-compiling

For a Linux arm64 binary to drop on a Spark itself:

```bash
GOOS=linux GOARCH=arm64 go build -o llm-dashboard .
```

## License

MIT.
