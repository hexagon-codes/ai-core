package streamx

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGeminiParserFunctionSnapshotsAndReportedZeroUsage(t *testing.T) {
	data := "data: " + `{"responseId":"response-original","modelVersion":"actual-model","candidates":[{"content":{"role":"model","parts":[{"text":"prefix"},{"functionCall":{"id":"call-original","name":"emit_schema","args":{"title":"draft"}}}]}}]}` + "\n\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call-original","name":"emit_schema","args":{"title":"synthetic","count":1000000000000000001}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}` + "\n\n"
	stream := NewStream(strings.NewReader(data), GeminiFormat)
	result, err := stream.Collect()
	_ = stream.Close()
	if err != nil || result.Content != "prefix" || result.ID != "response-original" || result.Model != "actual-model" || result.FinishReason != "STOP" || len(result.ToolCalls) != 1 || result.ToolCalls[0].ID != "call-original" || result.ToolCalls[0].Name != "emit_schema" {
		t.Fatalf("result=%+v error=%v", result, err)
	}
	var args struct {
		Title string      `json:"title"`
		Count json.Number `json:"count"`
	}
	if err := json.Unmarshal([]byte(result.ToolCalls[0].Arguments), &args); err != nil || args.Title != "synthetic" || args.Count.String() != "1000000000000000001" {
		t.Fatalf("complete args snapshots were concatenated or rounded: %q error=%v", result.ToolCalls[0].Arguments, err)
	}
	if result.UsageReceipt == nil || result.UsageReceipt.TotalTokens != 0 {
		t.Fatalf("explicit zero receipt lost: %+v", result)
	}
}

func TestGeminiParserSameNamedCallsHaveIndependentIdentity(t *testing.T) {
	data := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"same","args":{"value":"A"}}},{"functionCall":{"name":"same","args":{"value":"B"}}}]},"finishReason":"STOP"}]}` + "\n\n"
	stream := NewStream(strings.NewReader(data), GeminiFormat)
	result, err := stream.Collect()
	_ = stream.Close()
	if err != nil || len(result.ToolCalls) != 2 {
		t.Fatalf("result=%+v error=%v", result, err)
	}
	a, b := result.ToolCalls[0], result.ToolCalls[1]
	if a.ID == "" || b.ID == "" || a.ID == b.ID || a.Index == b.Index || a.Arguments != `{"value":"A"}` || b.Arguments != `{"value":"B"}` {
		t.Fatalf("independent calls were merged by name or part position: %+v", result.ToolCalls)
	}
	if result.UsageReceipt != nil {
		t.Fatal("absent usage was converted to a zero receipt")
	}
}
