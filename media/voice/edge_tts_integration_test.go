//go:build integration

// 真机集成测试：打微软真实免费 Edge TTS 端点，验证协议 + Sec-MS-GEC 版本仍被接受。
// 默认不跑（需网络且依赖微软非官方端点）；版本漂移会让此测试先红，提前预警。
//
//	go test -tags integration -run TestEdgeTTS_Integration ./media/voice/
package voice

import (
	"context"
	"testing"
	"time"
)

func TestEdgeTTS_Integration_RealEndpointReturnsMP3(t *testing.T) {
	tts := NewEdgeTTS()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := tts.Synthesize(ctx, "你好，我是河蟹。", SynthesizeOptions{})
	if err != nil {
		t.Fatalf("真实 Edge TTS 失败（可能是 Sec-MS-GEC-Version 过旧被微软拒绝，需同步 defaultEdgeSecMSGECVersion）: %v", err)
	}
	if res.Format != FormatMP3 {
		t.Errorf("格式应为 MP3，实际 %v", res.Format)
	}
	// MP3 magic：ID3 或帧同步 0xFF。
	if len(res.Audio) < 1000 || !(res.Audio[0] == 0xFF || string(res.Audio[:3]) == "ID3") {
		t.Errorf("返回内容不像 MP3：%d 字节，头 %x", len(res.Audio), res.Audio[:min(4, len(res.Audio))])
	}
	t.Logf("真实 Edge TTS OK：%d 字节 MP3", len(res.Audio))
}
