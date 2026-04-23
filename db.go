package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	_ "modernc.org/sqlite" // pure-Go SQLite driver, no CGO
)

// Record is one observed LLM request.
type Record struct {
	ID               int64             `json:"id"`
	Timestamp        time.Time         `json:"ts"`
	Task             string            `json:"task"`               // first ~120 chars of the prompt
	Model            string            `json:"model"`
	PromptTokens     int               `json:"prompt_tokens"`
	CompletionTokens int               `json:"completion_tokens"`
	TotalMS          int               `json:"total_ms"`           // wall-clock request duration
	PrefillMS        int               `json:"prefill_ms"`         // upstream -> first token (streaming only)
	DecodeMS         int               `json:"decode_ms"`          // first token -> last token (streaming only)
	TokensPerSecond  float64           `json:"tokens_per_second"`
	GPUTemps         map[string]float64 `json:"gpu_temps"`         // spark name -> peak temp °C during the request
	Status           int               `json:"status"`
}

// Spark is one GPU node we can SSH into.
type Spark struct {
	Name     string `json:"name"`
	Host     string `json:"host"`
	User     string `json:"user"`
	Port     int    `json:"port"`
	KeyPath  string `json:"key_path"`
	GPUIndex int    `json:"gpu_index"`
}

type Store struct {
	db *sql.DB
	mu sync.Mutex
}

func OpenStore(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, err
	}
	if err := db.Ping(); err != nil {
		return nil, err
	}
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) migrate() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS records(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts INTEGER NOT NULL,
			task TEXT,
			model TEXT,
			prompt_tokens INTEGER,
			completion_tokens INTEGER,
			total_ms INTEGER,
			prefill_ms INTEGER,
			decode_ms INTEGER,
			tokens_per_second REAL,
			gpu_temps TEXT,
			status INTEGER
		)`,
		`CREATE INDEX IF NOT EXISTS idx_records_ts ON records(ts DESC)`,
		`CREATE TABLE IF NOT EXISTS sparks(
			name TEXT PRIMARY KEY,
			host TEXT NOT NULL,
			user TEXT NOT NULL,
			port INTEGER NOT NULL DEFAULT 22,
			key_path TEXT,
			gpu_index INTEGER NOT NULL DEFAULT 0
		)`,
		`CREATE TABLE IF NOT EXISTS config(
			key TEXT PRIMARY KEY,
			value TEXT
		)`,
	}
	for _, q := range stmts {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("migrate: %w", err)
		}
	}
	return nil
}

func (s *Store) GetConfig(key string) (string, error) {
	var v string
	err := s.db.QueryRow(`SELECT value FROM config WHERE key=?`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return v, err
}

func (s *Store) SetConfig(key, value string) error {
	_, err := s.db.Exec(`INSERT INTO config(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`, key, value)
	return err
}

func (s *Store) ListSparks() ([]Spark, error) {
	rows, err := s.db.Query(`SELECT name,host,user,port,COALESCE(key_path,''),gpu_index FROM sparks ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Spark
	for rows.Next() {
		var sp Spark
		if err := rows.Scan(&sp.Name, &sp.Host, &sp.User, &sp.Port, &sp.KeyPath, &sp.GPUIndex); err != nil {
			return nil, err
		}
		out = append(out, sp)
	}
	return out, rows.Err()
}

func (s *Store) UpsertSpark(sp Spark) error {
	if sp.Port == 0 {
		sp.Port = 22
	}
	_, err := s.db.Exec(`INSERT INTO sparks(name,host,user,port,key_path,gpu_index) VALUES(?,?,?,?,?,?)
		ON CONFLICT(name) DO UPDATE SET host=excluded.host,user=excluded.user,port=excluded.port,key_path=excluded.key_path,gpu_index=excluded.gpu_index`,
		sp.Name, sp.Host, sp.User, sp.Port, sp.KeyPath, sp.GPUIndex)
	return err
}

func (s *Store) DeleteSpark(name string) error {
	_, err := s.db.Exec(`DELETE FROM sparks WHERE name=?`, name)
	return err
}

func (s *Store) InsertRecord(r *Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	temps, _ := json.Marshal(r.GPUTemps)
	res, err := s.db.Exec(`INSERT INTO records(ts,task,model,prompt_tokens,completion_tokens,total_ms,prefill_ms,decode_ms,tokens_per_second,gpu_temps,status)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)`,
		r.Timestamp.UnixMilli(), r.Task, r.Model, r.PromptTokens, r.CompletionTokens,
		r.TotalMS, r.PrefillMS, r.DecodeMS, r.TokensPerSecond, string(temps), r.Status)
	if err != nil {
		return err
	}
	r.ID, _ = res.LastInsertId()
	return nil
}

func (s *Store) RecentRecords(limit int) ([]Record, error) {
	if limit <= 0 {
		limit = 200
	}
	rows, err := s.db.Query(`SELECT id,ts,task,model,prompt_tokens,completion_tokens,total_ms,prefill_ms,decode_ms,tokens_per_second,gpu_temps,status
		FROM records ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Record
	for rows.Next() {
		var r Record
		var tsMs int64
		var temps string
		if err := rows.Scan(&r.ID, &tsMs, &r.Task, &r.Model, &r.PromptTokens, &r.CompletionTokens,
			&r.TotalMS, &r.PrefillMS, &r.DecodeMS, &r.TokensPerSecond, &temps, &r.Status); err != nil {
			return nil, err
		}
		r.Timestamp = time.UnixMilli(tsMs)
		if temps != "" {
			_ = json.Unmarshal([]byte(temps), &r.GPUTemps)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
