// Package llmmon instruments LLM HTTP clients (OpenAI, Anthropic, Gemini, etc.)
// by wrapping http.RoundTripper. Zero dependencies beyond stdlib.
//
// Usage:
//
//	client := &http.Client{Transport: llmmon.Wrap(llmmon.Options{
//	    Endpoint:   "http://macmon-server:8280",
//	    Token:      os.Getenv("MACMON_LLM_INGEST_TOKEN"),
//	    App:        "my-service",
//	    LogPrompts: true, // optional: log prompt/response bodies
//	})}
package llmmon

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Options configures the SDK.
type Options struct {
	// Endpoint is the macmon-server base URL (e.g. "http://localhost:8280").
	Endpoint string
	// Token authenticates telemetry ingestion. When empty, MACMON_LLM_INGEST_TOKEN is used.
	Token string
	// App is the application name tag sent with every call record.
	App string
	// Feature tags the product feature (e.g. "chat", "briefing"). Optional.
	Feature string
	// LogPrompts enables capturing prompt input and model response text.
	// Disabled by default for privacy. Enable only in non-production or with consent.
	LogPrompts bool
	// Transport is the underlying RoundTripper. nil = http.DefaultTransport.
	Transport http.RoundTripper
}

// Wrap returns an http.RoundTripper that intercepts LLM API calls and reports
// metrics to macmon-server asynchronously.
func Wrap(opts Options) http.RoundTripper {
	base := opts.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	ep := strings.TrimRight(opts.Endpoint, "/")
	token := opts.Token
	if token == "" {
		token = os.Getenv("MACMON_LLM_INGEST_TOKEN")
	}
	t := &interceptor{
		base:       base,
		app:        opts.App,
		feature:    opts.Feature,
		logPrompts: opts.LogPrompts,
		exporter:   newExporter(ep, token),
	}
	go t.cleanupRetries()
	return t
}

// NewClient returns a pre-wrapped *http.Client.
func NewClient(opts Options) *http.Client {
	return &http.Client{Transport: Wrap(opts)}
}

// ── interceptor ──────────────────────────────────────────────────────────────

type interceptor struct {
	base       http.RoundTripper
	app        string
	feature    string
	logPrompts bool
	exporter   *exporter
	retries    sync.Map // fingerprint(string) → *retryEntry
}

// retryEntry tracks how many times a request with the same fingerprint (method+path+body)
// has been seen recently — used to detect retries the underlying vendor SDK issues on its
// own (this transport has no retry logic itself, it only observes).
type retryEntry struct {
	mu       sync.Mutex
	count    int
	lastSeen time.Time
}

const retryWindow = 2 * time.Minute

// ── trace linking (opt-in) ───────────────────────────────────────────────────
//
// This transport has no knowledge of any APM agent. It only reads a W3C traceparent
// if one already reaches the outgoing request — either as an HTTP header (e.g. an APM
// agent's outbound-call instrumentation already injected it, as macmon-apm-java's OkHttp
// advice does today) or via an explicit context.Context value set through
// ContextWithTraceparent for callers that carry trace context in Go's context instead.

type traceparentCtxKey struct{}

// ContextWithTraceparent returns a context carrying an explicit W3C traceparent string
// ("00-{32 hex trace-id}-{16 hex parent-id}-{2 hex flags}"), for callers that don't have
// it as an outgoing HTTP header. The transport checks the header first, then this value.
func ContextWithTraceparent(ctx context.Context, traceparent string) context.Context {
	return context.WithValue(ctx, traceparentCtxKey{}, traceparent)
}

// extractTraceID pulls the trace-id segment out of a W3C traceparent, checking the
// request header first and falling back to an explicit context value. Empty if neither
// is present or malformed — this is best-effort, not a hard requirement.
func extractTraceID(req *http.Request) string {
	tp := req.Header.Get("traceparent")
	if tp == "" {
		if v, ok := req.Context().Value(traceparentCtxKey{}).(string); ok {
			tp = v
		}
	}
	if tp == "" {
		return ""
	}
	parts := strings.Split(tp, "-")
	if len(parts) < 2 || len(parts[1]) != 32 {
		return ""
	}
	return parts[1]
}

// requestFingerprint identifies "the same logical request" across retry attempts.
func requestFingerprint(req *http.Request, body []byte) string {
	h := sha256.New()
	h.Write([]byte(req.Method))
	h.Write([]byte(req.URL.Path))
	h.Write(body)
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// bumpRetry returns 0 for a request's first attempt, then 1, 2, ... for each subsequent
// attempt with the same fingerprint seen within retryWindow. Entries older than the window
// reset to 0 instead of accumulating forever (a genuinely new call can coincidentally repeat
// the same body, e.g. an identical prompt sent again later).
func (t *interceptor) bumpRetry(fp string) int {
	now := time.Now()
	v, loaded := t.retries.LoadOrStore(fp, &retryEntry{lastSeen: now})
	if !loaded {
		return 0
	}
	e := v.(*retryEntry)
	e.mu.Lock()
	defer e.mu.Unlock()
	if now.Sub(e.lastSeen) > retryWindow {
		e.count = 0
	} else {
		e.count++
	}
	e.lastSeen = now
	return e.count
}

// cleanupRetries evicts stale fingerprints so long-running processes don't leak memory
// across many distinct one-off requests.
func (t *interceptor) cleanupRetries() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		now := time.Now()
		t.retries.Range(func(k, v any) bool {
			e := v.(*retryEntry)
			e.mu.Lock()
			stale := now.Sub(e.lastSeen) > retryWindow
			e.mu.Unlock()
			if stale {
				t.retries.Delete(k)
			}
			return true
		})
	}
}

func (t *interceptor) RoundTrip(req *http.Request) (*http.Response, error) {
	provider := detectProvider(req.URL.Host)
	if provider == "" {
		return t.base.RoundTrip(req)
	}

	var reqBody []byte
	if req.Body != nil {
		reqBody, _ = io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(reqBody))
	}

	retryCount := t.bumpRetry(requestFingerprint(req, reqBody))
	traceID := extractTraceID(req)
	isStream := isStreamingReq(reqBody)
	start := time.Now()

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		rec := makeRecord(provider, reqBody, nil, 0, time.Since(start).Milliseconds(), 0, t.app, t.feature, t.logPrompts)
		rec.Error = err.Error()
		rec.RetryCount = retryCount
		rec.TraceID = traceID
		t.exporter.send(rec)
		return resp, err
	}

	if isStream && resp.StatusCode == 200 {
		// Wrap body to measure TTFT while streaming.
		wrapped := &streamingBody{
			rc:         resp.Body,
			start:      start,
			provider:   provider,
			reqBody:    reqBody,
			app:        t.app,
			feature:    t.feature,
			logPrompts: t.logPrompts,
			exporter:   t.exporter,
			status:     resp.StatusCode,
			retryCount: retryCount,
			traceID:    traceID,
		}
		resp.Body = wrapped
		return resp, nil
	}

	// Non-streaming: read full body, parse, report.
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body = io.NopCloser(bytes.NewReader(respBody))
	rec := makeRecord(provider, reqBody, respBody, resp.StatusCode, time.Since(start).Milliseconds(), 0, t.app, t.feature, t.logPrompts)
	rec.RetryCount = retryCount
	rec.TraceID = traceID
	t.exporter.send(rec)
	return resp, nil
}

// ── streaming body wrapper ───────────────────────────────────────────────────

type streamingBody struct {
	rc         io.ReadCloser
	start      time.Time
	ttftMs     int64
	firstChunk bool
	buf        bytes.Buffer
	provider   string
	reqBody    []byte
	app        string
	feature    string
	logPrompts bool
	exporter   *exporter
	status     int
	done       bool
	retryCount int
	traceID    string
}

func (s *streamingBody) Read(p []byte) (int, error) {
	n, err := s.rc.Read(p)
	if n > 0 {
		s.buf.Write(p[:n])
		if !s.firstChunk {
			s.firstChunk = true
			s.ttftMs = time.Since(s.start).Milliseconds()
		}
	}
	if err == io.EOF && !s.done {
		s.done = true
		latency := time.Since(s.start).Milliseconds()
		rec := parseStreamRecord(s.provider, s.reqBody, s.buf.Bytes(), s.status, latency, s.ttftMs, s.app, s.feature, s.logPrompts)
		rec.RetryCount = s.retryCount
		rec.TraceID = s.traceID
		s.exporter.send(rec)
	}
	return n, err
}

func (s *streamingBody) Close() error {
	if !s.done {
		s.done = true
		latency := time.Since(s.start).Milliseconds()
		rec := parseStreamRecord(s.provider, s.reqBody, s.buf.Bytes(), s.status, latency, s.ttftMs, s.app, s.feature, s.logPrompts)
		rec.RetryCount = s.retryCount
		rec.TraceID = s.traceID
		s.exporter.send(rec)
	}
	return s.rc.Close()
}

// ── provider detection ───────────────────────────────────────────────────────

func detectProvider(host string) string {
	switch {
	case strings.Contains(host, "api.openai.com"):
		return "openai"
	case strings.Contains(host, "api.anthropic.com"):
		return "anthropic"
	case strings.Contains(host, "generativelanguage.googleapis.com"):
		return "gemini"
	case strings.Contains(host, "openrouter.ai"):
		return "openrouter"
	case strings.Contains(host, "api.mistral.ai"):
		return "mistral"
	case strings.Contains(host, "api.cohere.com"):
		return "cohere"
	case strings.Contains(host, "api.groq.com"):
		return "groq"
	default:
		return ""
	}
}

func isStreamingReq(body []byte) bool {
	return bytes.Contains(body, []byte(`"stream":true`)) ||
		bytes.Contains(body, []byte(`"stream": true`))
}

// ── CallRecord ────────────────────────────────────────────────────────────────

// CallRecord is a single LLM API call record sent to macmon-server.
type CallRecord struct {
	Timestamp     string        `json:"timestamp"`
	Provider      string        `json:"provider"`
	Model         string        `json:"model"`
	App           string        `json:"app"`
	Feature       string        `json:"feature,omitempty"`
	PromptTok     int           `json:"prompt_tokens"`
	CompleteTok   int           `json:"completion_tokens"`
	TotalTok      int           `json:"total_tokens"`
	CacheReadTok  int           `json:"cache_read_tokens,omitempty"`
	CacheWriteTok int           `json:"cache_write_tokens,omitempty"`
	LatencyMs     int64         `json:"latency_ms"`
	TTFTMs        int64         `json:"ttft_ms,omitempty"`
	StatusCode    int           `json:"status_code"`
	CostUSD       float64       `json:"cost_usd"`
	Streaming     bool          `json:"streaming,omitempty"`
	ToolCalls     []ToolCallRec `json:"tool_calls,omitempty"`
	PromptText    string        `json:"prompt_text,omitempty"`
	ResponseText  string        `json:"response_text,omitempty"`
	Error         string        `json:"error,omitempty"`
	ErrorCode     string        `json:"error_code,omitempty"`
	RetryCount    int           `json:"retry_count,omitempty"`
	TraceID       string        `json:"trace_id,omitempty"`
}

// ToolCallRec records a single function/tool call within an LLM response.
type ToolCallRec struct {
	Name string `json:"name"`
	ID   string `json:"id,omitempty"`
}

// ── request/response parsing ─────────────────────────────────────────────────

type genericReq struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type genericResp struct {
	Model string `json:"model"`
	// OpenAI
	Usage   openaiUsage `json:"usage"`
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	// Anthropic
	AnthUsage anthropicUsage `json:"usage"`
	Content   []struct {
		Type string `json:"type"`
		Text string `json:"text"`
		Name string `json:"name"`
		ID   string `json:"id"`
	} `json:"content"`
	// Error: message는 모든 공급자 공통. Type은 OpenAI/Anthropic/Groq/Mistral/OpenRouter류(공통 REST 관례),
	// Status는 Gemini("RESOURCE_EXHAUSTED" 등) 전용 필드명이라 따로 둠 — 우선순위는 호출부에서 결정.
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type,omitempty"`
		Status  string `json:"status,omitempty"`
	} `json:"error,omitempty"`
}

func makeRecord(provider string, reqBody, respBody []byte, status int, latencyMs, ttftMs int64, app, feature string, logPrompts bool) CallRecord {
	rec := CallRecord{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Provider:   provider,
		App:        app,
		Feature:    feature,
		LatencyMs:  latencyMs,
		TTFTMs:     ttftMs,
		StatusCode: status,
	}

	var rq genericReq
	_ = json.Unmarshal(reqBody, &rq)
	rec.Model = rq.Model
	rec.Streaming = rq.Stream

	if logPrompts && len(rq.Messages) > 0 {
		rec.PromptText = extractPromptText(rq.Messages)
	}

	if len(respBody) > 0 {
		var rp genericResp
		if err := json.Unmarshal(respBody, &rp); err == nil {
			if rec.Model == "" && rp.Model != "" {
				rec.Model = rp.Model
			}
			// OpenAI tokens
			if rp.Usage.TotalTokens > 0 {
				rec.PromptTok = rp.Usage.PromptTokens
				rec.CompleteTok = rp.Usage.CompletionTokens
				rec.TotalTok = rp.Usage.TotalTokens
			}
			// Anthropic tokens
			if rp.AnthUsage.InputTokens > 0 || rp.AnthUsage.OutputTokens > 0 {
				rec.PromptTok = rp.AnthUsage.InputTokens
				rec.CompleteTok = rp.AnthUsage.OutputTokens
				rec.TotalTok = rp.AnthUsage.InputTokens + rp.AnthUsage.OutputTokens
				rec.CacheReadTok = rp.AnthUsage.CacheReadInputTokens
				rec.CacheWriteTok = rp.AnthUsage.CacheCreationInputTokens
			}
			// Tool calls (OpenAI)
			if len(rp.Choices) > 0 {
				for _, tc := range rp.Choices[0].Message.ToolCalls {
					rec.ToolCalls = append(rec.ToolCalls, ToolCallRec{Name: tc.Function.Name, ID: tc.ID})
				}
				if logPrompts && rp.Choices[0].Message.Content != "" {
					rec.ResponseText = rp.Choices[0].Message.Content
				}
			}
			// Tool calls / response (Anthropic)
			for _, blk := range rp.Content {
				switch blk.Type {
				case "tool_use":
					rec.ToolCalls = append(rec.ToolCalls, ToolCallRec{Name: blk.Name, ID: blk.ID})
				case "text":
					if logPrompts && blk.Text != "" {
						rec.ResponseText = blk.Text
					}
				}
			}
			if rp.Error != nil {
				rec.Error = rp.Error.Message
				if rp.Error.Type != "" {
					rec.ErrorCode = rp.Error.Type
				} else {
					rec.ErrorCode = rp.Error.Status
				}
			}
		}
	}

	rec.CostUSD = estimateCost(provider, rec.Model, rec.PromptTok, rec.CompleteTok, rec.CacheReadTok)
	return rec
}

// parseStreamRecord assembles a record from buffered SSE chunks.
func parseStreamRecord(provider string, reqBody, sseBody []byte, status int, latencyMs, ttftMs int64, app, feature string, logPrompts bool) CallRecord {
	rec := CallRecord{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Provider:   provider,
		App:        app,
		Feature:    feature,
		LatencyMs:  latencyMs,
		TTFTMs:     ttftMs,
		StatusCode: status,
		Streaming:  true,
	}

	var rq genericReq
	_ = json.Unmarshal(reqBody, &rq)
	rec.Model = rq.Model
	if logPrompts {
		rec.PromptText = extractPromptText(rq.Messages)
	}

	// Parse SSE stream: accumulate delta content + final usage chunk.
	var sb strings.Builder
	sc := bufio.NewScanner(bytes.NewReader(sseBody))
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk genericResp
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if rec.Model == "" && chunk.Model != "" {
			rec.Model = chunk.Model
		}
		// Accumulate text delta
		if len(chunk.Choices) > 0 {
			sb.WriteString(chunk.Choices[0].Delta.Content)
		}
		// Final usage chunk (OpenAI sends usage in last chunk when stream_options.include_usage=true)
		if chunk.Usage.TotalTokens > 0 {
			rec.PromptTok = chunk.Usage.PromptTokens
			rec.CompleteTok = chunk.Usage.CompletionTokens
			rec.TotalTok = chunk.Usage.TotalTokens
		}
		// Anthropic streaming usage (message_delta event)
		if chunk.AnthUsage.OutputTokens > 0 {
			rec.CompleteTok = chunk.AnthUsage.OutputTokens
		}
		if chunk.AnthUsage.InputTokens > 0 {
			rec.PromptTok = chunk.AnthUsage.InputTokens
			rec.CacheReadTok = chunk.AnthUsage.CacheReadInputTokens
			rec.CacheWriteTok = chunk.AnthUsage.CacheCreationInputTokens
		}
	}
	if rec.TotalTok == 0 {
		rec.TotalTok = rec.PromptTok + rec.CompleteTok
	}
	if logPrompts && sb.Len() > 0 {
		rec.ResponseText = sb.String()
	}
	rec.CostUSD = estimateCost(provider, rec.Model, rec.PromptTok, rec.CompleteTok, rec.CacheReadTok)
	return rec
}

func extractPromptText(msgs []struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}) string {
	var parts []string
	for _, m := range msgs {
		switch v := m.Content.(type) {
		case string:
			parts = append(parts, m.Role+": "+v)
		}
	}
	return strings.Join(parts, "\n")
}

// ── cost estimation ───────────────────────────────────────────────────────────

func estimateCost(provider, model string, promptTok, completeTok, cacheReadTok int) float64 {
	in, out := pricePer1M(provider, model)
	if in == 0 && out == 0 {
		return 0
	}
	cost := float64(promptTok)/1_000_000*in + float64(completeTok)/1_000_000*out
	// Cache read tokens are billed at ~10% of input price (Anthropic)
	if cacheReadTok > 0 {
		cost += float64(cacheReadTok) / 1_000_000 * in * 0.1
	}
	return cost
}

// pricePer1M returns (input, output) USD per 1M tokens.
func pricePer1M(provider, model string) (float64, float64) {
	m := strings.ToLower(model)
	switch provider {
	case "openai":
		switch {
		case strings.Contains(m, "gpt-4o-mini"):
			return 0.15, 0.60
		case strings.Contains(m, "gpt-4o"):
			return 2.50, 10.00
		case strings.Contains(m, "gpt-4-turbo"):
			return 10.00, 30.00
		case strings.Contains(m, "gpt-4"):
			return 30.00, 60.00
		case strings.Contains(m, "gpt-3.5"):
			return 0.50, 1.50
		case strings.Contains(m, "o1-mini"):
			return 3.00, 12.00
		case strings.Contains(m, "o1"):
			return 15.00, 60.00
		case strings.Contains(m, "o3-mini"):
			return 1.10, 4.40
		case strings.Contains(m, "o3"):
			return 10.00, 40.00
		}
	case "anthropic":
		switch {
		case strings.Contains(m, "haiku-4-5") || strings.Contains(m, "haiku-3-5"):
			return 0.80, 4.00
		case strings.Contains(m, "haiku"):
			return 0.25, 1.25
		case strings.Contains(m, "sonnet-4"):
			return 3.00, 15.00
		case strings.Contains(m, "sonnet-3-7"), strings.Contains(m, "sonnet-3.7"):
			return 3.00, 15.00
		case strings.Contains(m, "sonnet-3-5"), strings.Contains(m, "sonnet-3.5"):
			return 3.00, 15.00
		case strings.Contains(m, "sonnet"):
			return 3.00, 15.00
		case strings.Contains(m, "opus-4"):
			return 15.00, 75.00
		case strings.Contains(m, "opus"):
			return 15.00, 75.00
		}
	case "gemini":
		switch {
		case strings.Contains(m, "flash"):
			return 0.075, 0.30
		case strings.Contains(m, "pro"):
			return 1.25, 5.00
		case strings.Contains(m, "ultra"):
			return 7.00, 21.00
		}
	case "mistral":
		switch {
		case strings.Contains(m, "tiny"):
			return 0.25, 0.25
		case strings.Contains(m, "small"):
			return 1.00, 3.00
		case strings.Contains(m, "medium"):
			return 2.70, 8.10
		case strings.Contains(m, "large"):
			return 8.00, 24.00
		}
	case "groq":
		return 0.10, 0.10
	case "cohere":
		switch {
		case strings.Contains(m, "r7b"):
			return 0.0375, 0.15
		case strings.Contains(m, "r-plus"), strings.Contains(m, "r+"):
			return 2.50, 10.00
		case strings.Contains(m, "command-a"):
			return 2.50, 10.00
		case strings.Contains(m, "command-r"):
			return 0.50, 1.50
		case strings.Contains(m, "light"):
			return 0.30, 0.60
		}
	case "openrouter":
		// OpenRouter는 자체 마진 없이 하위 모델 가격을 그대로 패스스루 —
		// "vendor/model" 형태의 모델명에서 vendor를 떼어 기존 표를 재사용한다.
		if vendor, rest, found := strings.Cut(m, "/"); found {
			switch vendor {
			case "openai":
				return pricePer1M("openai", rest)
			case "anthropic":
				return pricePer1M("anthropic", rest)
			case "google":
				return pricePer1M("gemini", rest)
			case "mistralai", "mistral":
				return pricePer1M("mistral", rest)
			case "cohere":
				return pricePer1M("cohere", rest)
			}
		}
	}
	return 0, 0
}
