package voice

import (
	"strings"
	"testing"
)

func TestReadVoiceBodyRejectsOversizedResponse(t *testing.T) {
	t.Parallel()

	if _, err := readVoiceBody(strings.NewReader("12345"), 4, "test-tts"); err == nil {
		t.Fatal("readVoiceBody() accepted a response larger than the limit")
	}
	data, err := readVoiceBody(strings.NewReader("1234"), 4, "test-tts")
	if err != nil {
		t.Fatalf("readVoiceBody() error = %v", err)
	}
	if string(data) != "1234" {
		t.Fatalf("readVoiceBody() = %q, want %q", data, "1234")
	}
}

func TestURLPathEscapeEscapesQueryAndFragment(t *testing.T) {
	t.Parallel()

	escaped := urlPathEscape("voice?x=1#frag ment")
	if strings.ContainsAny(escaped, "?# ") {
		t.Fatalf("urlPathEscape() = %q, query/fragment/space remained unescaped", escaped)
	}
}
