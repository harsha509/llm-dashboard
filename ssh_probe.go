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

// Probe samples GPU temperatures from each configured Spark via SSH. One
// Probe is shared by the whole app; the scraper calls ReadAll() once per tick.
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

// ReadAll takes one temperature reading from every configured Spark in
// parallel and returns a map keyed by Spark.Name. Sparks we can't reach
// are omitted from the result; the reason is logged once per tick so an
// empty tile on the dashboard has a corresponding line in the terminal.
func (p *Probe) ReadAll(ctx context.Context) map[string]float64 {
	sparks, err := p.store.ListSparks()
	if err != nil {
		log.Printf("probe: list sparks: %v", err)
		return nil
	}
	if len(sparks) == 0 {
		return nil
	}
	out := make(map[string]float64, len(sparks))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, sp := range sparks {
		wg.Add(1)
		go func(sp Spark) {
			defer wg.Done()
			// readOnce ignores ctx; this gives us a ceiling anyway.
			done := make(chan struct{})
			var t float64
			var readErr error
			go func() {
				t, readErr = p.readOnce(sp)
				close(done)
			}()
			select {
			case <-done:
			case <-ctx.Done():
				log.Printf("probe: %s: cancelled before response", sp.Name)
				return
			}
			if readErr != nil {
				log.Printf("probe: %s (%s@%s:%d gpu=%d): %v",
					sp.Name, sp.User, sp.Host, sp.Port, sp.GPUIndex, readErr)
				return
			}
			mu.Lock()
			out[sp.Name] = t
			mu.Unlock()
		}(sp)
	}
	wg.Wait()
	return out
}

// readOnce runs `nvidia-smi --query-gpu=temperature.gpu -i <idx> --format=csv,noheader,nounits`
// on the target spark over SSH. Returns the parsed temperature or the
// underlying error (dial/auth/exec/parse) so callers can surface it.
func (p *Probe) readOnce(sp Spark) (float64, error) {
	cli, err := p.clientFor(sp)
	if err != nil {
		return 0, fmt.Errorf("ssh dial: %w", err)
	}
	sess, err := cli.NewSession()
	if err != nil {
		// Connection probably went stale — drop it so next call reconnects.
		p.drop(sp.Name)
		return 0, fmt.Errorf("ssh new session (will reconnect next tick): %w", err)
	}
	defer sess.Close()

	cmd := fmt.Sprintf("nvidia-smi --query-gpu=temperature.gpu -i %d --format=csv,noheader,nounits", sp.GPUIndex)
	var out, errBuf bytes.Buffer
	sess.Stdout = &out
	sess.Stderr = &errBuf
	if err := sess.Run(cmd); err != nil {
		stderr := strings.TrimSpace(errBuf.String())
		if stderr != "" {
			return 0, fmt.Errorf("%s: %v (stderr: %s)", cmd, err, stderr)
		}
		return 0, fmt.Errorf("%s: %w", cmd, err)
	}
	s := strings.TrimSpace(out.String())
	if s == "" {
		return 0, fmt.Errorf("%s returned empty output", cmd)
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %q: %w", s, err)
	}
	return v, nil
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
