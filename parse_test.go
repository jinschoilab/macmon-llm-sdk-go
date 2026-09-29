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
