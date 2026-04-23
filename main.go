// llm-dashboard: passive monitoring dashboard for locally-hosted vLLM / OpenAI-
// compatible LLM servers. Sits in front of your LLM as a reverse proxy, records
// tokens/second from every request, and correlates it with GPU temperature
// sampled from DGX Spark nodes over SSH.
//
// Two HTTP listeners run side-by-side:
//   -ui   : the dashboard (default :7002)   -> browse here
//   -proxy: the LLM proxy   (default :7001) -> point your clients here
//
// Point your LLM clients (OpenAI SDK, curl, etc.) at the proxy instead of the
// upstream vLLM directly. The proxy forwards transparently and records metrics
// out-of-band.
//
// Note on :7002 default: macOS Monterey+ uses :7000 for AirPlay Receiver
// (running under ControlCenter), which is relaunched by launchd if killed.
// We pick :7002 so a fresh install works without System Settings changes.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("%v", err)
	}
}

func run() error {
	var (
		uiAddr    = flag.String("ui", ":7002", "dashboard UI listen address")
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
		return fmt.Errorf("open db: %w", err)
	}
	defer store.Close()
	absDB, err := filepath.Abs(*dbPath)
	if err != nil {
		absDB = *dbPath // fall back to what the user passed
	}
	log.Printf("database   -> %s", absDB)

	probe := NewProbe(store)

	proxy, err := NewProxy(store, probe)
	if err != nil {
		return fmt.Errorf("build proxy: %w", err)
	}

	uiMux := http.NewServeMux()
	RegisterUI(uiMux, store, proxy)

	proxyMux := http.NewServeMux()
	proxyMux.Handle("/", proxy)

	// Bind both listeners up front. If either fails we bail out cleanly — no
	// half-started state, and defer store.Close() above still runs.
	uiLn, err := net.Listen("tcp", *uiAddr)
	if err != nil {
		return bindError("ui", "-ui", *uiAddr, err)
	}
	proxyLn, err := net.Listen("tcp", *proxyAddr)
	if err != nil {
		_ = uiLn.Close()
		return bindError("proxy", "-proxy", *proxyAddr, err)
	}

	uiSrv := &http.Server{Handler: withAccessLog(uiMux), ReadHeaderTimeout: 5 * time.Second}
	proxySrv := &http.Server{Handler: proxyMux, ReadHeaderTimeout: 10 * time.Second}

	srvErr := make(chan error, 2)
	go func() {
		log.Printf("dashboard  -> http://localhost%s", *uiAddr)
		if err := uiSrv.Serve(uiLn); err != nil && !errors.Is(err, http.ErrServerClosed) {
			srvErr <- fmt.Errorf("ui server: %w", err)
		}
	}()
	go func() {
		log.Printf("llm proxy  -> http://localhost%s  (forwards to configured upstream)", *proxyAddr)
		if err := proxySrv.Serve(proxyLn); err != nil && !errors.Is(err, http.ErrServerClosed) {
			srvErr <- fmt.Errorf("proxy server: %w", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-stop:
		log.Printf("received %s, shutting down", sig)
	case err := <-srvErr:
		log.Printf("server failed: %v — shutting down peer", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = uiSrv.Shutdown(ctx)
	_ = proxySrv.Shutdown(ctx)
	return nil
}

// bindError formats a listen failure with a hint at which flag to change.
func bindError(name, flagName, addr string, err error) error {
	var se *net.OpError
	if errors.As(err, &se) && se.Err != nil {
		// Surface the useful part (e.g. "bind: address already in use")
		// along with the flag hint, so the user sees the fix in the error.
		return fmt.Errorf("%s listener failed to bind %s (%v) — change it with %s <addr>, e.g. %s :7003",
			name, addr, se.Err, flagName, flagName)
	}
	return fmt.Errorf("%s listener failed to bind %s: %v (change with %s <addr>)", name, addr, err, flagName)
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
