// llm-dashboard: passive monitoring dashboard for locally-hosted vLLM / OpenAI-
// compatible LLM servers. Sits in front of your LLM as a reverse proxy, records
// tokens/second from every request, and correlates it with GPU temperature
// sampled from DGX Spark nodes over SSH.
//
// Two HTTP listeners run side-by-side:
//   -ui   : the dashboard (default :7000)   -> browse here
//   -proxy: the LLM proxy   (default :7001) -> point your clients here
//
// Point your LLM clients (OpenAI SDK, curl, etc.) at the proxy instead of the
// upstream vLLM directly. The proxy forwards transparently and records metrics
// out-of-band.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	var (
		uiAddr    = flag.String("ui", ":7000", "dashboard UI listen address")
		proxyAddr = flag.String("proxy", ":7001", "LLM reverse-proxy listen address")
		dbPath    = flag.String("db", "", "SQLite path (default: ./llm-dashboard.db)")
	)
	flag.Parse()

	if *dbPath == "" {
		wd, _ := os.Getwd()
		*dbPath = filepath.Join(wd, "llm-dashboard.db")
	}

	store, err := OpenStore(*dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer store.Close()

	probe := NewProbe(store)

	proxy, err := NewProxy(store, probe)
	if err != nil {
		log.Fatalf("build proxy: %v", err)
	}

	uiMux := http.NewServeMux()
	RegisterUI(uiMux, store, proxy)

	proxyMux := http.NewServeMux()
	proxyMux.Handle("/", proxy)

	uiSrv := &http.Server{Addr: *uiAddr, Handler: withAccessLog(uiMux), ReadHeaderTimeout: 5 * time.Second}
	proxySrv := &http.Server{Addr: *proxyAddr, Handler: proxyMux, ReadHeaderTimeout: 10 * time.Second}

	go func() {
		log.Printf("dashboard  -> http://localhost%s", *uiAddr)
		if err := uiSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("ui server: %v", err)
		}
	}()
	go func() {
		log.Printf("llm proxy -> http://localhost%s  (forwards to configured upstream)", *proxyAddr)
		if err := proxySrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("proxy server: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	log.Println("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = uiSrv.Shutdown(ctx)
	_ = proxySrv.Shutdown(ctx)
}

func withAccessLog(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		h.ServeHTTP(w, r)
		if r.URL.Path != "/api/records" { // suppress the 2s poll spam
			log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(t0))
		}
	})
}
