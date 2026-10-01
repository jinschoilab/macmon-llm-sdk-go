package llmmon

import "testing"

// Regression: genericResp once had two fields tagged `json:"usage"`, which makes
// encoding/json drop both — every token count came out as 0.
func TestMakeRecordParsesUsage(t *testing.T) {
	anth := makeRecord("anthropic", []byte(`{"model":"claude-opus-5-5","messages":[]}`),
		[]byte(`{"model":"claude-opus-5-5","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}`),
		200, 50, 0, "app", "", false)
	if anth.PromptTok != 10 || anth.CompleteTok != 5 || anth.TotalTok != 15 || anth.CacheReadTok != 100 || anth.CacheWriteTok != 20 {
		t.Fatalf("anthropic usage not parsed: %+v", anth)
	}
	oa := makeRecord("openai", []byte(`{"model":"gpt-4o","messages":[]}`),
		[]byte(`{"model":"gpt-4o","choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`),
		200, 50, 0, "app", "", false)
	if oa.PromptTok != 7 || oa.CompleteTok != 3 || oa.TotalTok != 10 {
		t.Fatalf("openai usage not parsed: %+v", oa)
	}
	stream := parseStreamRecord("anthropic", []byte(`{"model":"claude-opus-5-5","stream":true}`),
		[]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42,\"input_tokens\":9}}\n\n"),
		200, 80, 30, "app", "", false)
	if stream.CompleteTok != 42 || stream.PromptTok != 9 {
		t.Fatalf("stream usage not parsed: %+v", stream)
	}
}

// Regression: Anthropic streaming puts input/cache tokens in message_start's message.usage,
// not at the top level — they used to be recorded as 0 (cost under-reported).
func TestParseStreamRecordMessageStartUsage(t *testing.T) {
	sse := "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":120,\"cache_read_input_tokens\":300,\"cache_creation_input_tokens\":40,\"output_tokens\":1}}}\n\n" +
		"event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n"
	rec := parseStreamRecord("anthropic", []byte(`{"stream":true}`), []byte(sse), 200, 80, 30, "app", "", false)
	if rec.PromptTok != 120 || rec.CacheReadTok != 300 || rec.CacheWriteTok != 40 || rec.CompleteTok != 42 {
		t.Fatalf("message_start usage not parsed: %+v", rec)
	}
	if rec.Model != "claude-opus-5-5" {
		t.Fatalf("model not taken from message_start: %q", rec.Model)
	}
}
