package llmmon

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExporterSendsIngestToken(t *testing.T) {
	auth := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	exporter := newExporter(server.URL, "ingest-secret")
	exporter.buf = []CallRecord{{Provider: "test", Model: "test-model"}}
	exporter.flush()

	if got := <-auth; got != "Bearer ingest-secret" {
		t.Fatalf("Authorization = %q, want Bearer ingest-secret", got)
	}
}
