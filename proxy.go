package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Proxy is the passive sniffer. It forwards requests to the configured upstream
// vLLM / OpenAI-compatible server and records per-request metrics out-of-band.
type Proxy struct {
	store *Store
	probe *Probe
	http  *http.Client
}

func NewProxy(store *Store, probe *Probe) (*Proxy, error) {
	return &Proxy{
		store: store,
		probe: probe,
		http: &http.Client{
			Timeout: 0, // let the LLM take as long as it wants
			Transport: &http.Transport{
				// We need Transport.ResponseHeaderTimeout=0 so streaming works.
				DisableCompression: true,
				MaxIdleConns:       20,
				IdleConnTimeout:    90 * time.Second,
			},
		},
	}, nil
}

func (p *Proxy) upstreamURL() (*url.URL, error) {
	raw, err := p.store.GetConfig("upstream_url")
	if err != nil {
		return nil, err
	}
	if raw == "" {
		return nil, fmt.Errorf("no upstream_url configured — set it in the dashboard Settings panel")
	}
	return url.Parse(strings.TrimRight(raw, "/"))
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	upstream, err := p.upstreamURL()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// Read the full request body so we can inspect it AND forward it.
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read request: "+err.Error(), http.StatusBadRequest)
		return
	}
	_ = r.Body.Close()

	// Best-effort parse of OpenAI-style request.
	reqMeta := parseRequestBody(r.URL.Path, reqBody)

	// Build outbound request
	target := *upstream
	target.Path = singleJoin(upstream.Path, r.URL.Path)
	target.RawQuery = r.URL.RawQuery

	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bytes.NewReader(reqBody))
	if err != nil {
		http.Error(w, "build upstream req: "+err.Error(), http.StatusInternalServerError)
		return
	}
	copyHeader(outReq.Header, r.Header)
	outReq.Header.Del("Accept-Encoding") // we disabled compression; keep the body readable

	// Start GPU sampling only for endpoints we care about — skip for /v1/models etc.
	var stopProbe func() map[string]float64
	shouldRecord := isGeneration(r.URL.Path)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	if shouldRecord {
		stopProbe = p.probe.Sample(ctx)
	}

	t0 := time.Now()
	resp, err := p.http.Do(outReq)
	if err != nil {
		if stopProbe != nil {
			stopProbe()
		}
		http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	isStream := reqMeta.Stream || strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")

	rec := &Record{
		Timestamp: t0,
		Task:      summarize(reqMeta.Prompt, 120),
		Model:     reqMeta.Model,
		Status:    resp.StatusCode,
	}

	if isStream {
		p.relayStreaming(w, resp.Body, rec)
	} else {
		p.relayBuffered(w, resp.Body, rec)
	}

	rec.TotalMS = int(time.Since(t0).Milliseconds())
	if rec.CompletionTokens > 0 && rec.TotalMS > 0 {
		// If we have decode timing use it (more accurate), otherwise fall back
		// to wall-clock total.
		divisor := rec.DecodeMS
		if divisor <= 0 {
			divisor = rec.TotalMS
		}
		rec.TokensPerSecond = float64(rec.CompletionTokens) / (float64(divisor) / 1000.0)
	}

	if stopProbe != nil {
		rec.GPUTemps = stopProbe()
	}

	if shouldRecord && resp.StatusCode < 500 {
		if err := p.store.InsertRecord(rec); err != nil {
			log.Printf("record insert: %v", err)
		}
	}
}

// relayBuffered forwards a non-streaming response and parses the final JSON.
func (p *Proxy) relayBuffered(w http.ResponseWriter, body io.Reader, rec *Record) {
	// Tee to both client and our parser.
	var buf bytes.Buffer
	tee := io.TeeReader(body, &buf)
	if _, err := io.Copy(w, tee); err != nil {
		log.Printf("relay buffered: %v", err)
		return
	}
	parseCompletionJSON(buf.Bytes(), rec)
}

// relayStreaming forwards an SSE stream chunk-by-chunk while tracking first-
// token timing and extracting usage from the final chunk.
func (p *Proxy) relayStreaming(w http.ResponseWriter, body io.Reader, rec *Record) {
	flusher, _ := w.(http.Flusher)
	reader := bufio.NewReaderSize(body, 64*1024)

	start := time.Now()
	var firstToken time.Time
	var lastToken time.Time
	tokenCount := 0

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if _, werr := w.Write(line); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}

			// SSE lines look like: "data: {json}\n"
			trimmed := bytes.TrimSpace(line)
			if bytes.HasPrefix(trimmed, []byte("data:")) {
				payload := bytes.TrimSpace(trimmed[len("data:"):])
				if len(payload) > 0 && !bytes.Equal(payload, []byte("[DONE]")) {
					if usage := extractStreamChunk(payload); usage != nil {
						if usage.newText && firstToken.IsZero() {
							firstToken = time.Now()
						}
						if usage.newText {
							tokenCount++
							lastToken = time.Now()
						}
						if usage.promptTokens > 0 {
							rec.PromptTokens = usage.promptTokens
						}
						if usage.completionTokens > 0 {
							rec.CompletionTokens = usage.completionTokens
						}
					}
				}
			}
		}
		if err != nil {
			break
		}
	}

	if !firstToken.IsZero() {
		rec.PrefillMS = int(firstToken.Sub(start).Milliseconds())
		if !lastToken.IsZero() {
			rec.DecodeMS = int(lastToken.Sub(firstToken).Milliseconds())
		}
	}
	// Fall back to counting chunks if vLLM didn't send usage.
	if rec.CompletionTokens == 0 && tokenCount > 0 {
		rec.CompletionTokens = tokenCount
	}
}

// --- request / response parsing -------------------------------------------------

type reqMeta struct {
	Model  string
	Stream bool
	Prompt string
}

func parseRequestBody(path string, body []byte) reqMeta {
	var out reqMeta
	if len(body) == 0 {
		return out
	}
	// Try generic map parse — handles both /chat/completions and /completions.
	var v map[string]any
	if err := json.Unmarshal(body, &v); err != nil {
		return out
	}
	if m, ok := v["model"].(string); ok {
		out.Model = m
	}
	if s, ok := v["stream"].(bool); ok {
		out.Stream = s
	}
	if p, ok := v["prompt"].(string); ok {
		out.Prompt = p
	}
	if msgs, ok := v["messages"].([]any); ok {
		// Concatenate a short preview of the last user/system message.
		for i := len(msgs) - 1; i >= 0; i-- {
			if m, ok := msgs[i].(map[string]any); ok {
				if content, ok := m["content"].(string); ok {
					out.Prompt = content
					break
				}
			}
		}
	}
	return out
}

func parseCompletionJSON(body []byte, rec *Record) {
	var v struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &v); err != nil {
		return
	}
	if v.Model != "" && rec.Model == "" {
		rec.Model = v.Model
	}
	rec.PromptTokens = v.Usage.PromptTokens
	rec.CompletionTokens = v.Usage.CompletionTokens
}

type streamChunk struct {
	promptTokens     int
	completionTokens int
	newText          bool
}

func extractStreamChunk(payload []byte) *streamChunk {
	var v struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			Text string `json:"text"` // /completions streaming uses text
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &v); err != nil {
		return nil
	}
	ck := &streamChunk{}
	if v.Usage != nil {
		ck.promptTokens = v.Usage.PromptTokens
		ck.completionTokens = v.Usage.CompletionTokens
	}
	for _, c := range v.Choices {
		if c.Delta.Content != "" || c.Text != "" {
			ck.newText = true
			break
		}
	}
	return ck
}

func isGeneration(p string) bool {
	return strings.HasSuffix(p, "/v1/chat/completions") ||
		strings.HasSuffix(p, "/v1/completions") ||
		strings.HasSuffix(p, "/completion") // llama.cpp native
}

func copyHeader(dst, src http.Header) {
	for k, vs := range src {
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func singleJoin(a, b string) string {
	a = strings.TrimRight(a, "/")
	b = "/" + strings.TrimLeft(b, "/")
	return a + b
}

func summarize(s string, n int) string {
	s = strings.TrimSpace(s)
	// collapse whitespace
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n-1] + "…"
	}
	return s
}
