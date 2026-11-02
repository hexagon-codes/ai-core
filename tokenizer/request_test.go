package tokenizer

import (
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestEstimateCompletionRequestCountsSchemasAndArguments(t *testing.T) {
	base, err := EstimateCompletionRequest(llm.CompletionRequest{Messages: []llm.Message{llm.AssistantToolCallMessage("", []llm.ToolCallRef{{ID: "call", Name: "lookup"}})}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.CompletionRequest{
		Messages:       []llm.Message{llm.AssistantToolCallMessage("", []llm.ToolCallRef{{ID: "call", Name: "lookup", Arguments: `{"query":"` + strings.Repeat("large argument ", 100) + `"}`}})},
		Tools:          []llm.ToolDefinition{llm.NewToolDefinition("lookup", strings.Repeat("tool description ", 100), &llm.Schema{Type: "object", Properties: map[string]*llm.Schema{"query": {Type: "string", Description: strings.Repeat("query constraint ", 100)}}})},
		ResponseFormat: &llm.ResponseFormat{Type: "json_schema", JSONSchema: &llm.ResponseFormatJSONSchema{Name: "answer", Schema: &llm.Schema{Type: "object", Properties: map[string]*llm.Schema{"answer": {Type: "string"}}}}},
	}
	estimate, err := EstimateCompletionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if estimate.ToolSchemaTokens <= 0 || estimate.ResponseSchemaTokens <= 0 || estimate.ToolArgumentTokens <= 0 {
		t.Fatalf("context fields were omitted: %+v", estimate)
	}
	if estimate.MessageTokens <= base.MessageTokens || estimate.Tokens <= base.Tokens {
		t.Fatalf("tool arguments or schema did not increase context estimate: base=%+v estimate=%+v", base, estimate)
	}
	if estimate.Tokens != estimate.MessageTokens+estimate.ToolSchemaTokens+estimate.ResponseSchemaTokens+estimate.ControlTokens {
		t.Fatal("tool arguments were charged twice outside their message category")
	}
}

func TestEstimateCompletionRequestDeclaresUnknownMediaContext(t *testing.T) {
	request := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: strings.Repeat("ignored fallback ", 1000), MultiContent: []llm.ContentPart{{Type: "text", Text: "visible text"}, {Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64," + strings.Repeat("A", 10000), Detail: "high"}}}}}, Metadata: map[string]any{"provider_context_option": "declared"}}
	estimate, err := EstimateCompletionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	request.Messages[0].Content = "different ignored fallback"
	request.Messages[0].MultiContent[1].ImageURL.URL = "asset://image"
	short, err := EstimateCompletionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if estimate.UnestimatedMediaParts != 1 || !estimate.HasProviderMetadata {
		t.Fatalf("unknown media/provider context was reported as fully counted: %+v", estimate)
	}
	if estimate.Tokens != short.Tokens {
		t.Fatal("image transport bytes or ignored fallback text were counted as visual tokens")
	}
}
