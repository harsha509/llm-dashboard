// llm-dashboard: passive monitoring dashboard for locally-hosted vLLM /
// OpenAI-compatible LLM servers. Polls the upstream vLLM /metrics endpoint
// on a timer, samples GPU temperature from each configured DGX Spark node
// over SSH, and renders a live view correlating the two.
//
// There is no reverse proxy. Clients talk to vLLM directly at the upstream
// URL configured in Settings; the dashboard just observes what vLLM already
// reports.
//
// Flags:
//
//	-ui <addr>  dashboard UI listen address (default :7002)
//	-db <path>  SQLite path (default: ./llm-dashboard.db)
//	-interval   /metrics scrape interval (default 5s)
//
// Note on :7002 default: macOS Monterey+ uses :7000 for AirPlay Receiver
// (under ControlCenter), which launchd respawns if killed — we pick :7002
// so a fresh install works without System Settings changes.
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
		uiAddr   = flag.String("ui", ":7002", "dashboard UI listen address")
		dbPath   = flag.String("db", "", "SQLite path (default: ./llm-dashboard.db)")
		interval = flag.Duration("interval", 5*time.Second, "/metrics scrape interval")
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
		absDB = *dbPath
	}
	log.Printf("database   -> %s", absDB)

	probe := NewProbe(store)
	scraper := NewScraper(store, probe, *interval)

	uiMux := http.NewServeMux()
	RegisterUI(uiMux, store)

	// Bind UI listener up front so a conflict fails cleanly before we start
	// any goroutines — defer store.Close() above still runs.
	uiLn, err := net.Listen("tcp", *uiAddr)
	if err != nil {
		return bindError("ui", "-ui", *uiAddr, err)
	}

	uiSrv := &http.Server{Handler: withAccessLog(uiMux), ReadHeaderTimeout: 5 * time.Second}

	// Scraper runs until the main context cancels.
	scrapeCtx, cancelScrape := context.WithCancel(context.Background())
	defer cancelScrape()
	go scraper.Run(scrapeCtx)

	srvErr := make(chan error, 1)
	go func() {
		log.Printf("dashboard  -> http://localhost%s", *uiAddr)
		if err := uiSrv.Serve(uiLn); err != nil && !errors.Is(err, http.ErrServerClosed) {
			srvErr <- fmt.Errorf("ui server: %w", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-stop:
		log.Printf("received %s, shutting down", sig)
	case err := <-srvErr:
		log.Printf("server failed: %v — shutting down", err)
	}

	cancelScrape()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = uiSrv.Shutdown(ctx)
	return nil
}

// bindError formats a listen failure with a hint at which flag to change.
func bindError(name, flagName, addr string, err error) error {
	var se *net.OpError
	if errors.As(err, &se) && se.Err != nil {
		return fmt.Errorf("%s listener failed to bind %s (%v) — change it with %s <addr>, e.g. %s :7003",
			name, addr, se.Err, flagName, flagName)
	}
	return fmt.Errorf("%s listener failed to bind %s: %v (change with %s <addr>)", name, addr, err, flagName)
}

func withAccessLog(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		h.ServeHTTP(w, r)
		// /api/samples is polled every 2s by the UI; don't spam the log.
		if r.URL.Path != "/api/samples" {
			log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(t0))
		}
	})
}
