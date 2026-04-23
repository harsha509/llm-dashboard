package main

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
)

//go:embed web
var webFS embed.FS

// RegisterUI wires the dashboard's HTTP handlers onto mux.
func RegisterUI(mux *http.ServeMux, store *Store, _ *Proxy) {
	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("/", http.FileServer(http.FS(sub)))

	mux.HandleFunc("/api/records", func(w http.ResponseWriter, r *http.Request) {
		recs, err := store.RecentRecords(500)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, map[string]any{"records": recs})
	})

	mux.HandleFunc("/api/config", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			upstream, _ := store.GetConfig("upstream_url")
			sparks, _ := store.ListSparks()
			writeJSON(w, map[string]any{"upstream_url": upstream, "sparks": sparks})
		case http.MethodPost:
			var body struct {
				UpstreamURL string  `json:"upstream_url"`
				Sparks      []Spark `json:"sparks"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			if err := store.SetConfig("upstream_url", body.UpstreamURL); err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			// Replace-all semantics for the sparks list.
			existing, _ := store.ListSparks()
			keep := map[string]bool{}
			for _, sp := range body.Sparks {
				if sp.Name == "" {
					continue
				}
				keep[sp.Name] = true
				if err := store.UpsertSpark(sp); err != nil {
					http.Error(w, err.Error(), 500)
					return
				}
			}
			for _, sp := range existing {
				if !keep[sp.Name] {
					_ = store.DeleteSpark(sp.Name)
				}
			}
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
