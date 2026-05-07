// Package llmmon instruments LLM HTTP clients (OpenAI, Anthropic, Gemini, etc.)
// by wrapping http.RoundTripper. Zero dependencies beyond stdlib.
//
// Usage:
//
//	client := &http.Client{Transport: llmmon.Wrap(llmmon.Options{
//	    Endpoint:   "http://macmon-server:8280",
//	    App:        "my-service",
//	    LogPrompts: true, // optional: log prompt/response bodies
//	})}
package llmmon

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// Options configures the SDK.
type Options struct {
	// Endpoint is the macmon-server base URL (e.g. "http://localhost:8280").
	Endpoint string
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
	return &interceptor{
		base:       base,
		app:        opts.App,
		feature:    opts.Feature,
		logPrompts: opts.LogPrompts,
		exporter:   newExporter(ep),
	}
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

	isStream := isStreamingReq(reqBody)
	start := time.Now()

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		rec := makeRecord(provider, reqBody, nil, 0, time.Since(start).Milliseconds(), 0, t.app, t.feature, t.logPrompts)
		rec.Error = err.Error()
		t.exporter.send(rec)
		return resp, err
	}

	if isStream && resp.StatusCode == 200 {
		// Wrap body to measure TTFT while streaming.
		wrapped := &streamingBody{
			rc:       resp.Body,
			start:    start,
			provider: provider,
			reqBody:  reqBody,
			app:      t.app,
			feature:  t.feature,
			logPrompts: t.logPrompts,
			exporter: t.exporter,
			status:   resp.StatusCode,
		}
		resp.Body = wrapped
		return resp, nil
	}

	// Non-streaming: read full body, parse, report.
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body = io.NopCloser(bytes.NewReader(respBody))
	rec := makeRecord(provider, reqBody, respBody, resp.StatusCode, time.Since(start).Milliseconds(), 0, t.app, t.feature, t.logPrompts)
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
		s.exporter.send(rec)
	}
	return n, err
}

func (s *streamingBody) Close() error {
	if !s.done {
		s.done = true
		latency := time.Since(s.start).Milliseconds()
		rec := parseStreamRecord(s.provider, s.reqBody, s.buf.Bytes(), s.status, latency, s.ttftMs, s.app, s.feature, s.logPrompts)
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
	Timestamp       string        `json:"timestamp"`
	Provider        string        `json:"provider"`
	Model           string        `json:"model"`
	App             string        `json:"app"`
	Feature         string        `json:"feature,omitempty"`
	PromptTok       int           `json:"prompt_tokens"`
	CompleteTok     int           `json:"completion_tokens"`
	TotalTok        int           `json:"total_tokens"`
	CacheReadTok    int           `json:"cache_read_tokens,omitempty"`
	CacheWriteTok   int           `json:"cache_write_tokens,omitempty"`
	LatencyMs       int64         `json:"latency_ms"`
	TTFTMs          int64         `json:"ttft_ms,omitempty"`
	StatusCode      int           `json:"status_code"`
	CostUSD         float64       `json:"cost_usd"`
	Streaming       bool          `json:"streaming,omitempty"`
	ToolCalls       []ToolCallRec `json:"tool_calls,omitempty"`
	PromptText      string        `json:"prompt_text,omitempty"`
	ResponseText    string        `json:"response_text,omitempty"`
	Error           string        `json:"error,omitempty"`
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
	Model   string `json:"model"`
	// OpenAI
	Usage   openaiUsage `json:"usage"`
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct{ Name string `json:"name"` } `json:"function"`
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
	Error *struct {
		Message string `json:"message"`
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
	}
	return 0, 0
}
