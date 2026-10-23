package image

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIImageRequestUsesSupportedFormatAndStyleFields(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewNRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	for _, model := range []string{"gpt-image-2", "gpt-image-2.5-flare", "dall-e-3"} {
		t.Run(model, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body["model"] != model {
					t.Error("explicit model changed")
				}
				if strings.HasPrefix(model, "gpt-image-") {
					if _, present := body["response_format"]; present {
						t.Error("GPT Image does not accept legacy response_format")
					}
					if _, present := body["style"]; present {
						t.Error("GPT Image style belongs in the prompt")
					}
					if !strings.Contains(body["prompt"].(string), "vivid") {
						t.Error("requested style was lost")
					}
				} else if body["response_format"] != "b64_json" || body["style"] != "vivid" {
					t.Error("legacy adapter request changed")
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(encoded.Bytes())}}})
			}))
			defer server.Close()
			provider := NewOpenAICompatible("fixture", "fixture-key", server.URL, []string{model})
			if _, err := provider.Generate(context.Background(), Request{Model: model, Prompt: "a cat", Style: "vivid"}); err != nil {
				t.Fatal(err)
			}
		})
	}
}
