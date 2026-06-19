package voice

// 编译期接口断言：确保各具体 Provider 实现满足 TTS/STT 接口契约。
// 接口若发生漂移，将在此处直接编译失败。
var (
	_ TTSProvider = (*AzureTTS)(nil)
	_ TTSProvider = (*EdgeTTS)(nil)
	_ TTSProvider = (*OpenAITTS)(nil)
	_ TTSProvider = (*MiniMaxTTS)(nil)
	_ TTSProvider = (*ChainedTTS)(nil)

	_ STTProvider = (*AzureSTT)(nil)
	_ STTProvider = (*OpenAISTT)(nil)
)
