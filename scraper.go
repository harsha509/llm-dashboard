package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Scraper polls the configured upstream vLLM /metrics endpoint every
// ScrapeInterval, parses the Prometheus text payload for the vllm:* metrics
// we care about, samples GPU temps on the same tick, and writes one Sample
// row per scrape.
//
// The scrape target comes from Store.GetConfig("upstream_url") — the same
// field the old proxy used — so changing it in Settings takes effect on
// the next tick without restarting the process.
type Scraper struct {
	store    *Store
	probe    *Probe
	interval time.Duration
	http     *http.Client
}

func NewScraper(store *Store, probe *Probe, interval time.Duration) *Scraper {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	return &Scraper{
		store:    store,
		probe:    probe,
		interval: interval,
		http: &http.Client{
			// vLLM's /metrics endpoint can block briefly while the engine is
			// busy serving a request. 4s was too tight and produced spurious
			// "err" rows under load — 8s is still well under the 5s scrape
			// interval + safety margin.
			Timeout: 8 * time.Second,
		},
	}
}

// Run loops until ctx is cancelled. Errors are logged and do not stop the
// loop — a bad scrape just writes a Sample with ScrapeError populated so
// the UI can show an outage tile instead of going silent.
func (s *Scraper) Run(ctx context.Context) {
	log.Printf("scraper    -> polling every %s", s.interval)
	// Fire once immediately so the UI isn't empty for the first interval.
	s.tick(ctx)
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.tick(ctx)
		}
	}
}

func (s *Scraper) tick(ctx context.Context) {
	upstream, err := s.store.GetConfig("upstream_url")
	if err != nil {
		log.Printf("scraper: read upstream_url: %v", err)
		return
	}
	upstream = strings.TrimRight(strings.TrimSpace(upstream), "/")
	if upstream == "" {
		// Nothing to scrape yet — user hasn't configured the upstream.
		// Don't even insert a row; wait quietly.
		return
	}

	sample := &Sample{
		Timestamp: time.Now(),
		Upstream:  upstream,
	}

	// /metrics fetch — the main signal we care about.
	metrics, err := s.fetchMetrics(ctx, upstream+"/metrics")
	if err != nil {
		sample.ScrapeError = err.Error()
		log.Printf("scraper: fetch %s/metrics: %v", upstream, err)
	} else {
		s.populateFromMetrics(metrics, sample)
	}

	// /v1/models gives us the *actual* model path (e.g. "Qwen/Qwen3.6-35B-A3B")
	// even when the server was started with --served-model-name <alias>. vLLM
	// puts the real path in the "root" field of each model entry; we fall
	// back to "id" only when root is empty.
	if m := s.fetchModelName(ctx, upstream+"/v1/models"); m != "" {
		sample.Model = m
	}

	// GPU temps in parallel with /metrics would be nicer, but this is cheap
	// enough to do serially; ReadAll is already parallel across sparks.
	sample.GPUTemps = s.probe.ReadAll(ctx)

	if err := s.store.InsertSample(sample); err != nil {
		log.Printf("scraper: insert sample: %v", err)
	}
}

func (s *Scraper) fetchMetrics(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("http %d", resp.StatusCode)
	}
	// /metrics payloads are small (tens of KB); cap at 1 MB for safety.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// fetchModelName calls vLLM's OpenAI-compatible /v1/models and returns the
// underlying model path (HuggingFace ID or local path) for the first entry,
// preferring the "root" field when the server was launched with
// --served-model-name <alias>. Returns "" on any error; the scraper treats
// an empty model as "keep whatever we had last time" rather than clearing it.
func (s *Scraper) fetchModelName(ctx context.Context, url string) string {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return ""
	}
	resp, err := s.http.Do(req)
	if err != nil {
		log.Printf("scraper: fetch %s: %v", url, err)
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		log.Printf("scraper: fetch %s: http %d", url, resp.StatusCode)
		return ""
	}
	var v struct {
		Data []struct {
			ID   string `json:"id"`
			Root string `json:"root"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&v); err != nil {
		log.Printf("scraper: decode %s: %v", url, err)
		return ""
	}
	if len(v.Data) == 0 {
		return ""
	}
	// Prefer the "root" field — that's the HuggingFace path / local dir vLLM
	// was actually started from. Fall back to "id" (served-model-name) if
	// root is missing, which is what older vLLM versions may return.
	if v.Data[0].Root != "" {
		return v.Data[0].Root
	}
	return v.Data[0].ID
}

// populateFromMetrics fills a Sample from the Prometheus text payload. We
// only care about a handful of vllm:* series; anything else is ignored.
//
// vLLM metric names (stable across recent versions):
//   vllm:prompt_tokens_total                 (counter)
//   vllm:generation_tokens_total             (counter)
//   vllm:num_requests_running                (gauge)
//   vllm:num_requests_waiting                (gauge)
//   vllm:gpu_cache_usage_perc                (gauge, 0..1)
//   vllm:time_to_first_token_seconds_sum     (counter, histogram sum)
//   vllm:time_to_first_token_seconds_count   (counter, histogram count)
//   vllm:time_per_output_token_seconds_sum   (counter)
//   vllm:time_per_output_token_seconds_count (counter)
//   vllm:e2e_request_latency_seconds_sum     (counter)
//   vllm:e2e_request_latency_seconds_count   (counter)
//
// Each metric line looks like: `name{labels...} value` or `name value`.
// Multiple timeseries per metric (one per label set) get summed — a dash-
// board for a single upstream doesn't care about per-label breakdowns.
func (s *Scraper) populateFromMetrics(payload string, out *Sample) {
	sumInt := map[string]int64{}
	sumFloat := map[string]float64{}
	lastFloat := map[string]float64{} // for gauges: last value wins

	for _, raw := range strings.Split(payload, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, value, ok := splitMetricLine(line)
		if !ok {
			continue
		}
		switch name {
		case
			"vllm:prompt_tokens_total",
			"vllm:generation_tokens_total",
			"vllm:time_to_first_token_seconds_count",
			"vllm:time_per_output_token_seconds_count",
			"vllm:e2e_request_latency_seconds_count":
			// Integer counters — sum across labelsets.
			if v, err := strconv.ParseFloat(value, 64); err == nil {
				sumInt[name] += int64(v)
			}
		case
			"vllm:time_to_first_token_seconds_sum",
			"vllm:time_per_output_token_seconds_sum",
			"vllm:e2e_request_latency_seconds_sum":
			if v, err := strconv.ParseFloat(value, 64); err == nil {
				sumFloat[name] += v
			}
		case
			"vllm:num_requests_running",
			"vllm:num_requests_waiting",
			"vllm:gpu_cache_usage_perc":
			// Gauges — take the last one we see. For multi-label gauges we'd
			// want per-label but there's only one series here in practice.
			if v, err := strconv.ParseFloat(value, 64); err == nil {
				lastFloat[name] = v
			}
		}
	}

	out.PromptTokensTotal = sumInt["vllm:prompt_tokens_total"]
	out.GenerationTokensTotal = sumInt["vllm:generation_tokens_total"]
	out.TTFTSum = sumFloat["vllm:time_to_first_token_seconds_sum"]
	out.TTFTCount = sumInt["vllm:time_to_first_token_seconds_count"]
	out.TPOTSum = sumFloat["vllm:time_per_output_token_seconds_sum"]
	out.TPOTCount = sumInt["vllm:time_per_output_token_seconds_count"]
	out.E2ESum = sumFloat["vllm:e2e_request_latency_seconds_sum"]
	out.E2ECount = sumInt["vllm:e2e_request_latency_seconds_count"]
	out.NumRequestsRunning = lastFloat["vllm:num_requests_running"]
	out.NumRequestsWaiting = lastFloat["vllm:num_requests_waiting"]
	out.GPUCacheUsagePerc = lastFloat["vllm:gpu_cache_usage_perc"]
}

// splitMetricLine splits a Prometheus metric line into (name, value). The
// name is everything up to the first `{` or whitespace; the value is the
// final whitespace-separated token (a trailing timestamp, if present, is
// discarded by taking the field just before the end).
//
// We only need numeric extraction — we don't parse labels here because our
// metric-name switch in populateFromMetrics already filters to the series
// we care about and sums across all labelsets.
func splitMetricLine(line string) (name, value string, ok bool) {
	// Find end of metric name.
	idx := strings.IndexAny(line, "{ \t")
	if idx <= 0 {
		return "", "", false
	}
	name = line[:idx]
	// The value is the last whitespace-separated field, OR the last field
	// before a trailing timestamp. Prom format allows an optional trailing
	// timestamp after the value.
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return "", "", false
	}
	// With a timestamp: ["metric{..}", "value", "ts"] -> value is [-2]
	// Without:         ["metric{..}", "value"]        -> value is [-1]
	// Heuristic: if the last field parses as an int timestamp > 1e12, take [-2].
	last := fields[len(fields)-1]
	if _, err := strconv.ParseInt(last, 10, 64); err == nil && len(last) >= 13 {
		value = fields[len(fields)-2]
	} else {
		value = last
	}
	return name, value, true
}
