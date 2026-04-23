package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"io"
	"io/fs"
	"log"
	"net/http"
)

//go:embed web
var webFS embed.FS

// RegisterUI wires the dashboard's HTTP handlers onto mux. The scraper runs
// in its own goroutine in main; RegisterUI only needs read access to the
// store for rendering and write access for Settings edits.
func RegisterUI(mux *http.ServeMux, store *Store) {
	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("/", http.FileServer(http.FS(sub)))

	mux.HandleFunc("/api/samples", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			samples, err := store.RecentSamples(300)
			if err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			writeJSON(w, map[string]any{"samples": samples})
		case http.MethodDelete:
			if err := store.ClearSamples(); err != nil {
				log.Printf("samples DELETE: %v", err)
				http.Error(w, err.Error(), 500)
				return
			}
			log.Printf("samples DELETE: cleared")
			w.WriteHeader(204)
		default:
			http.Error(w, "method not allowed", 405)
		}
	})

	mux.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			upstream, _ := store.GetConfig("upstream_url")
			sparks, _ := store.ListSparks()
			writeJSON(w, map[string]any{
				"upstream_url": upstream,
				"sparks":       sparks,
			})
		case http.MethodPost:
			raw, err := io.ReadAll(r.Body)
			if err != nil {
				log.Printf("config POST: read body: %v", err)
				http.Error(w, "read body: "+err.Error(), 400)
				return
			}
			log.Printf("config POST: body=%s", string(raw))

			var body struct {
				UpstreamURL string  `json:"upstream_url"`
				Sparks      []Spark `json:"sparks"`
			}
			if err := json.NewDecoder(bytes.NewReader(raw)).Decode(&body); err != nil {
				log.Printf("config POST: decode: %v", err)
				http.Error(w, "decode: "+err.Error(), 400)
				return
			}
			log.Printf("config POST: upstream=%q sparks=%d", body.UpstreamURL, len(body.Sparks))

			if err := store.SetConfig("upstream_url", body.UpstreamURL); err != nil {
				log.Printf("config POST: set upstream: %v", err)
				http.Error(w, "set upstream: "+err.Error(), 500)
				return
			}
			// Replace-all semantics for the sparks list.
			existing, listErr := store.ListSparks()
			if listErr != nil {
				log.Printf("config POST: list existing sparks: %v", listErr)
			}
			keep := map[string]bool{}
			for _, sp := range body.Sparks {
				if sp.Name == "" {
					log.Printf("config POST: skipping spark with empty name (host=%q)", sp.Host)
					continue
				}
				keep[sp.Name] = true
				if err := store.UpsertSpark(sp); err != nil {
					log.Printf("config POST: upsert spark %q: %v", sp.Name, err)
					http.Error(w, "upsert "+sp.Name+": "+err.Error(), 500)
					return
				}
				log.Printf("config POST: upsert spark name=%q host=%q user=%q port=%d gpu=%d ok", sp.Name, sp.Host, sp.User, sp.Port, sp.GPUIndex)
			}
			for _, sp := range existing {
				if !keep[sp.Name] {
					if err := store.DeleteSpark(sp.Name); err != nil {
						log.Printf("config POST: delete stale spark %q: %v", sp.Name, err)
					} else {
						log.Printf("config POST: deleted stale spark %q", sp.Name)
					}
				}
			}
			log.Printf("config POST: done, %d sparks saved", len(keep))
			w.WriteHeader(204)
		default:
			http.Error(w, "method not allowed", 405)
		}
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
