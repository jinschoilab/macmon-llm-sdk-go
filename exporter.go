package llmmon

import (
	"bytes"
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// exporter batches CallRecords and ships them to macmon-server asynchronously.
type exporter struct {
	endpoint string
	token    string
	mu       sync.Mutex
	buf      []CallRecord
	client   *http.Client
	once     sync.Once
}

func newExporter(endpoint, token string) *exporter {
	e := &exporter{
		endpoint: endpoint,
		token:    token,
		client:   &http.Client{Timeout: 5 * time.Second},
	}
	return e
}

func (e *exporter) send(rec CallRecord) {
	// Start background flusher on first use.
	e.once.Do(func() {
		go e.loop()
	})
	e.mu.Lock()
	e.buf = append(e.buf, rec)
	e.mu.Unlock()
}

func (e *exporter) loop() {
	tick := time.NewTicker(3 * time.Second)
	defer tick.Stop()
	for range tick.C {
		e.flush()
	}
}

func (e *exporter) flush() {
	e.mu.Lock()
	if len(e.buf) == 0 {
		e.mu.Unlock()
		return
	}
	batch := e.buf
	e.buf = nil
	e.mu.Unlock()

	body, err := json.Marshal(map[string]any{"records": batch})
	if err != nil {
		return
	}
	req, err := http.NewRequest(http.MethodPost, e.endpoint+"/ingest/llm", bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if e.token != "" {
		req.Header.Set("Authorization", "Bearer "+e.token)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return
	}
	resp.Body.Close()
}
