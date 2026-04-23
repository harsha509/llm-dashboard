package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

// Probe samples GPU temperatures from each configured Spark during the window
// of an LLM request. One Probe is shared by the whole app.
type Probe struct {
	store *Store

	mu      sync.Mutex
	clients map[string]*sshClient // keyed by spark.Name
}

type sshClient struct {
	spark Spark
	conn  *ssh.Client
}

func NewProbe(store *Store) *Probe {
	return &Probe{store: store, clients: map[string]*sshClient{}}
}

// Sample returns the peak GPU temperature seen on each configured Spark between
// now and when stop() is called (or ctx expires). Call Sample() right before
// the upstream request and call the returned stop func when the response ends.
//
// The returned map has one entry per Spark; missing entries mean the probe
// could not reach that node and the dashboard will render "--".
func (p *Probe) Sample(ctx context.Context) (stop func() map[string]float64) {
	sparks, err := p.store.ListSparks()
	if err != nil {
		log.Printf("probe: list sparks: %v", err)
		return func() map[string]float64 { return nil }
	}
	if len(sparks) == 0 {
		return func() map[string]float64 { return nil }
	}

	peaks := make(map[string]float64, len(sparks))
	var mu sync.Mutex
	done := make(chan struct{})

	var wg sync.WaitGroup
	for _, sp := range sparks {
		wg.Add(1)
		go func(sp Spark) {
			defer wg.Done()
			tick := time.NewTicker(500 * time.Millisecond)
			defer tick.Stop()

			// Take an immediate reading so even short requests capture a value.
			if t, ok := p.readOnce(sp); ok {
				mu.Lock()
				if t > peaks[sp.Name] {
					peaks[sp.Name] = t
				}
				mu.Unlock()
			}

			for {
				select {
				case <-done:
					return
				case <-ctx.Done():
					return
				case <-tick.C:
					if t, ok := p.readOnce(sp); ok {
						mu.Lock()
						if t > peaks[sp.Name] {
							peaks[sp.Name] = t
						}
						mu.Unlock()
					}
				}
			}
		}(sp)
	}

	return func() map[string]float64 {
		close(done)
		wg.Wait()
		mu.Lock()
		defer mu.Unlock()
		out := make(map[string]float64, len(peaks))
		for k, v := range peaks {
			out[k] = v
		}
		return out
	}
}

// readOnce runs `nvidia-smi --query-gpu=temperature.gpu -i <idx> --format=csv,noheader,nounits`
// and returns the parsed integer temperature.
func (p *Probe) readOnce(sp Spark) (float64, bool) {
	cli, err := p.clientFor(sp)
	if err != nil {
		// Log at debug volume — SSH failures are common (key missing, node down)
		// and we don't want to spam the log on every tick.
		return 0, false
	}
	sess, err := cli.NewSession()
	if err != nil {
		// Connection probably went stale — drop it so next call reconnects.
		p.drop(sp.Name)
		return 0, false
	}
	defer sess.Close()

	cmd := fmt.Sprintf("nvidia-smi --query-gpu=temperature.gpu -i %d --format=csv,noheader,nounits", sp.GPUIndex)
	var out bytes.Buffer
	sess.Stdout = &out
	if err := sess.Run(cmd); err != nil {
		return 0, false
	}
	s := strings.TrimSpace(out.String())
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func (p *Probe) clientFor(sp Spark) (*ssh.Client, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if c, ok := p.clients[sp.Name]; ok && c.conn != nil {
		return c.conn, nil
	}
	cfg, err := sshConfig(sp)
	if err != nil {
		return nil, err
	}
	addr := net.JoinHostPort(sp.Host, strconv.Itoa(sp.Port))
	conn, err := ssh.Dial("tcp", addr, cfg)
	if err != nil {
		return nil, err
	}
	p.clients[sp.Name] = &sshClient{spark: sp, conn: conn}
	return conn, nil
}

func (p *Probe) drop(name string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if c, ok := p.clients[name]; ok {
		if c.conn != nil {
			_ = c.conn.Close()
		}
		delete(p.clients, name)
	}
}

func sshConfig(sp Spark) (*ssh.ClientConfig, error) {
	var auths []ssh.AuthMethod
	path := sp.KeyPath
	if path == "" {
		// Try the usual suspects.
		home, _ := os.UserHomeDir()
		for _, name := range []string{"id_ed25519", "id_rsa"} {
			candidate := home + "/.ssh/" + name
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
				break
			}
		}
	}
	if path != "" {
		if key, err := os.ReadFile(path); err == nil {
			if signer, err := ssh.ParsePrivateKey(key); err == nil {
				auths = append(auths, ssh.PublicKeys(signer))
			}
		}
	}
	if len(auths) == 0 {
		return nil, fmt.Errorf("no usable SSH key for spark %q (tried %q)", sp.Name, path)
	}
	return &ssh.ClientConfig{
		User:            sp.User,
		Auth:            auths,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // dashboard use, LAN; document this in README
		Timeout:         4 * time.Second,
	}, nil
}
