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
	SessionID        string            `json:"session_id"`         // derived from first user message + model; stable across turns
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

// Sample is one periodic scrape of the upstream vLLM /metrics endpoint,
// plus GPU temps from each configured Spark at that same tick. Rates
// (tokens/sec, queue deltas) are computed on the frontend from consecutive
// samples, so we store raw counters here.
type Sample struct {
	ID                    int64              `json:"id"`
	Timestamp             time.Time          `json:"ts"`
	Upstream              string             `json:"upstream"`                // scrape target, for when we scrape multiple
	Model                 string             `json:"model"`                   // served model name, from the model_name="..." label on vllm:* metrics
	PromptTokensTotal     int64              `json:"prompt_tokens_total"`     // vllm:prompt_tokens_total (counter)
	GenerationTokensTotal int64              `json:"generation_tokens_total"` // vllm:generation_tokens_total (counter)
	NumRequestsRunning    float64            `json:"num_requests_running"`    // vllm:num_requests_running (gauge)
	NumRequestsWaiting    float64            `json:"num_requests_waiting"`    // vllm:num_requests_waiting (gauge)
	GPUCacheUsagePerc     float64            `json:"gpu_cache_usage_perc"`    // vllm:gpu_cache_usage_perc (gauge, 0..1)
	TTFTSum               float64            `json:"ttft_sum"`                // vllm:time_to_first_token_seconds_sum
	TTFTCount             int64              `json:"ttft_count"`              // vllm:time_to_first_token_seconds_count
	TPOTSum               float64            `json:"tpot_sum"`                // vllm:time_per_output_token_seconds_sum
	TPOTCount             int64              `json:"tpot_count"`              // vllm:time_per_output_token_seconds_count
	E2ESum                float64            `json:"e2e_sum"`                 // vllm:e2e_request_latency_seconds_sum
	E2ECount              int64              `json:"e2e_count"`               // vllm:e2e_request_latency_seconds_count
	GPUTemps              map[string]float64 `json:"gpu_temps"`               // spark name -> °C at this tick
	ScrapeError           string             `json:"scrape_error"`            // non-empty if /metrics fetch or parse failed
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
			session_id TEXT,
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
		// samples is the time-series table for periodic /metrics scrapes.
		// One row per scrape tick (every few seconds). We store raw counter
		// and gauge values; the UI computes rates from consecutive rows.
		`CREATE TABLE IF NOT EXISTS samples(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts INTEGER NOT NULL,
			upstream TEXT,
			model TEXT,
			prompt_tokens_total INTEGER,
			generation_tokens_total INTEGER,
			num_requests_running REAL,
			num_requests_waiting REAL,
			gpu_cache_usage_perc REAL,
			ttft_sum REAL,
			ttft_count INTEGER,
			tpot_sum REAL,
			tpot_count INTEGER,
			e2e_sum REAL,
			e2e_count INTEGER,
			gpu_temps TEXT,
			scrape_error TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts DESC)`,
	}
	for _, q := range stmts {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("migrate: %w", err)
		}
	}

	// session_id was added in a later revision — add it to existing DBs.
	if err := s.addColumnIfMissing("records", "session_id", "TEXT"); err != nil {
		return fmt.Errorf("migrate session_id: %w", err)
	}
	if _, err := s.db.Exec(`CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id)`); err != nil {
		return fmt.Errorf("migrate session index: %w", err)
	}
	// model column on samples — added after the initial samples table landed.
	if err := s.addColumnIfMissing("samples", "model", "TEXT"); err != nil {
		return fmt.Errorf("migrate samples.model: %w", err)
	}
	return nil
}

// addColumnIfMissing runs ALTER TABLE ADD COLUMN only if the column is absent.
// Uses pragma_table_info so it's safe to call on fresh and upgraded DBs.
func (s *Store) addColumnIfMissing(table, column, decl string) error {
	var n int
	err := s.db.QueryRow(
		`SELECT COUNT(*) FROM pragma_table_info(?) WHERE name=?`,
		table, column,
	).Scan(&n)
	if err != nil {
		return err
	}
	if n > 0 {
		return nil
	}
	_, err = s.db.Exec(fmt.Sprintf(`ALTER TABLE %s ADD COLUMN %s %s`, table, column, decl))
	return err
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
	res, err := s.db.Exec(`INSERT INTO records(ts,session_id,task,model,prompt_tokens,completion_tokens,total_ms,prefill_ms,decode_ms,tokens_per_second,gpu_temps,status)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`,
		r.Timestamp.UnixMilli(), r.SessionID, r.Task, r.Model, r.PromptTokens, r.CompletionTokens,
		r.TotalMS, r.PrefillMS, r.DecodeMS, r.TokensPerSecond, string(temps), r.Status)
	if err != nil {
		return err
	}
	r.ID, _ = res.LastInsertId()
	return nil
}

// ClearSamples wipes the samples time-series AND the legacy records table
// (kept from the old proxy era). Called by the "Clear history" button in
// Settings.
func (s *Store) ClearSamples() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.db.Exec(`DELETE FROM samples`); err != nil {
		return err
	}
	// Old table from the proxy era; still worth clearing so "clear history"
	// actually means "fresh slate" even if someone reads the DB directly.
	if _, err := s.db.Exec(`DELETE FROM records`); err != nil {
		return err
	}
	// Reclaim space. VACUUM is a light operation on a small file.
	if _, err := s.db.Exec(`VACUUM`); err != nil {
		return err
	}
	return nil
}

func (s *Store) InsertSample(r *Sample) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	temps, _ := json.Marshal(r.GPUTemps)
	res, err := s.db.Exec(`INSERT INTO samples(
		ts, upstream, model,
		prompt_tokens_total, generation_tokens_total,
		num_requests_running, num_requests_waiting, gpu_cache_usage_perc,
		ttft_sum, ttft_count, tpot_sum, tpot_count, e2e_sum, e2e_count,
		gpu_temps, scrape_error
	) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		r.Timestamp.UnixMilli(), r.Upstream, r.Model,
		r.PromptTokensTotal, r.GenerationTokensTotal,
		r.NumRequestsRunning, r.NumRequestsWaiting, r.GPUCacheUsagePerc,
		r.TTFTSum, r.TTFTCount, r.TPOTSum, r.TPOTCount, r.E2ESum, r.E2ECount,
		string(temps), r.ScrapeError)
	if err != nil {
		return err
	}
	r.ID, _ = res.LastInsertId()
	return nil
}

func (s *Store) RecentSamples(limit int) ([]Sample, error) {
	if limit <= 0 {
		limit = 300
	}
	rows, err := s.db.Query(`SELECT id, ts, COALESCE(upstream,''), COALESCE(model,''),
		COALESCE(prompt_tokens_total,0), COALESCE(generation_tokens_total,0),
		COALESCE(num_requests_running,0), COALESCE(num_requests_waiting,0),
		COALESCE(gpu_cache_usage_perc,0),
		COALESCE(ttft_sum,0), COALESCE(ttft_count,0),
		COALESCE(tpot_sum,0), COALESCE(tpot_count,0),
		COALESCE(e2e_sum,0), COALESCE(e2e_count,0),
		COALESCE(gpu_temps,''), COALESCE(scrape_error,'')
		FROM samples ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Sample
	for rows.Next() {
		var r Sample
		var tsMs int64
		var temps string
		if err := rows.Scan(&r.ID, &tsMs, &r.Upstream, &r.Model,
			&r.PromptTokensTotal, &r.GenerationTokensTotal,
			&r.NumRequestsRunning, &r.NumRequestsWaiting, &r.GPUCacheUsagePerc,
			&r.TTFTSum, &r.TTFTCount, &r.TPOTSum, &r.TPOTCount, &r.E2ESum, &r.E2ECount,
			&temps, &r.ScrapeError); err != nil {
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

func (s *Store) RecentRecords(limit int) ([]Record, error) {
	if limit <= 0 {
		limit = 200
	}
	rows, err := s.db.Query(`SELECT id,ts,COALESCE(session_id,''),task,model,prompt_tokens,completion_tokens,total_ms,prefill_ms,decode_ms,tokens_per_second,gpu_temps,status
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
		if err := rows.Scan(&r.ID, &tsMs, &r.SessionID, &r.Task, &r.Model, &r.PromptTokens, &r.CompletionTokens,
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
