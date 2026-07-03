package streamx

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
)

var (
	// ErrStreamClosed 表示流已关闭，无法继续读取
	ErrStreamClosed = errors.New("streamx: stream closed")
	// ErrUnsupportedFormat 表示不支持的流式响应格式
	ErrUnsupportedFormat = errors.New("streamx: unsupported format")
)

// Format 定义流式响应的数据格式类型
// 不同的 AI 厂商使用不同的流式响应格式，本类型用于标识使用哪种解析器
type Format int

const (
	// OpenAIFormat OpenAI 流式格式
	// 使用 Server-Sent Events (SSE)，每行以 "data: " 前缀开始
	// 结束标记为 "data: [DONE]"
	OpenAIFormat Format = iota

	// ClaudeFormat Anthropic Claude 流式格式
	// 使用 SSE，包含多种事件类型：message_start、content_block_delta、message_stop 等
	ClaudeFormat

	// GeminiFormat Google Gemini 流式格式
	// 使用 JSON 数组格式，每个元素包含 candidates 数组
	GeminiFormat

	// CustomFormat 自定义格式
	// 需要配合 SetParser 方法使用自定义解析器
	CustomFormat
)

// Chunk 表示流式响应中的单个数据块
// 每次从流中读取数据时，会解析为一个 Chunk 对象
// 多个 Chunk 的 Content 拼接后形成完整的响应内容
type Chunk struct {
	// ID 响应的唯一标识符，通常在首个块中返回
	ID string `json:"id,omitempty"`
	// Content 本次增量的文本内容
	// 流式响应会将完整内容拆分为多个增量，每个块包含一部分
	Content string `json:"content,omitempty"`
	// Reasoning 推理/思考过程的增量内容
	// 支持 OpenAI o1/o3、DeepSeek-R1、Qwen3 等模型的 reasoning_content/reasoning 字段
	Reasoning string `json:"reasoning,omitempty"`
	// Role 消息角色，通常为 "assistant"
	// 一般只在首个块中包含此字段
	Role string `json:"role,omitempty"`
	// Model 使用的模型名称，如 "gpt-4"、"claude-3-opus" 等
	Model string `json:"model,omitempty"`
	// FinishReason 结束原因
	// 可能的值：stop（正常结束）、length（达到长度限制）、tool_calls（需要调用工具）等
	FinishReason string `json:"finish_reason,omitempty"`
	// ToolCalls 工具调用列表
	// 当模型决定调用工具时，此字段包含工具调用信息
	// 工具调用的参数可能分散在多个块中，需要合并
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// Index 多选项时的索引号
	// 当请求 n>1 时，用于区分不同的生成结果
	Index int `json:"index,omitempty"`
	// Raw 原始 JSON 数据
	// 保留原始数据以便需要时进行自定义解析
	Raw json.RawMessage `json:"raw,omitempty"`
}

// ToolCall 表示模型发起的工具/函数调用
// 在 Function Calling 场景中，模型可能请求调用外部工具
type ToolCall struct {
	// Index 流式响应中 tool_call 的位置索引（OpenAI 协议标准合并键）
	// OpenAI streaming 规范：仅首个 delta 携带 id/name/type，后续 delta 仅含 arguments 增量，
	// 必须按 Index 合并 delta；Claude/DeepSeek-R1/Qwen3 等会把 arguments 切成多个 delta。
	Index int `json:"index"`
	// ID 工具调用的唯一标识符
	// 用于在后续响应中匹配工具调用结果
	ID string `json:"id,omitempty"`
	// Type 工具类型，通常为 "function"
	Type string `json:"type,omitempty"`
	// Name 要调用的函数/工具名称
	Name string `json:"name,omitempty"`
	// Arguments 函数参数的 JSON 字符串
	// 在流式响应中，参数可能分多个块传输，需要拼接
	Arguments string `json:"arguments,omitempty"`
}

// Usage 记录本次请求的 Token 使用统计
// 用于计费和配额管理
type Usage struct {
	// PromptTokens 输入/提示词消耗的 Token 数
	PromptTokens int `json:"prompt_tokens,omitempty"`
	// CompletionTokens 输出/生成内容消耗的 Token 数
	CompletionTokens int `json:"completion_tokens,omitempty"`
	// TotalTokens 总计消耗的 Token 数（输入+输出）
	//
	// 注意：为保持向后兼容，TotalTokens 仍为 PromptTokens+CompletionTokens，
	// 不把缓存 Token 计入其中。缓存读/写 Token 由下面两个维度独立记录，
	// 需要"真实提示词总量"的消费方可自行计算
	// PromptTokens+CacheCreationTokens+CacheReadTokens。
	TotalTokens int `json:"total_tokens,omitempty"`
	// CacheCreationTokens 写入提示词缓存所消耗的 Token 数
	//
	// 对应 Anthropic 的 cache_creation_input_tokens。表示本次请求为建立
	// 提示词缓存而额外计费的输入 Token（首次写入缓存时产生）。
	// 不支持提示词缓存或本次未写缓存的 Provider 该值为 0。
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
	// CacheReadTokens 命中提示词缓存、从缓存读取的 Token 数
	//
	// 对应 Anthropic 的 cache_read_input_tokens、OpenAI 的
	// prompt_tokens_details.cached_tokens。表示本次请求命中缓存、
	// 以折扣价计费的输入 Token。未命中缓存时该值为 0。
	CacheReadTokens int `json:"cache_read_tokens,omitempty"`
}

// Result 表示流式响应处理完成后的完整结果
// 包含所有块合并后的完整内容和统计信息
type Result struct {
	// ID 响应的唯一标识符
	ID string `json:"id,omitempty"`
	// Content 所有块拼接后的完整文本内容
	Content string `json:"content,omitempty"`
	// Role 消息角色
	Role string `json:"role,omitempty"`
	// Model 使用的模型名称
	Model string `json:"model,omitempty"`
	// FinishReason 最终的结束原因
	FinishReason string `json:"finish_reason,omitempty"`
	// ToolCalls 合并后的完整工具调用列表
	// 工具调用的参数已从多个块中合并完成
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// Usage Token 使用统计（如果 API 返回）
	Usage Usage `json:"usage,omitempty"`
	// Chunks 保存所有原始块，用于调试或重放
	Chunks []*Chunk `json:"chunks,omitempty"`
}

// Stream 是流式响应的核心处理器
// 负责从 io.Reader 读取数据，解析为 Chunk，并提供多种消费方式
//
// 支持三种使用模式：
//  1. 通道模式：通过 Chunks() 返回的通道逐个接收块
//  2. 回调模式：通过 OnChunk/OnDone/OnError 设置回调函数
//  3. 收集模式：通过 Collect() 阻塞等待并收集完整结果
type Stream struct {
	reader  *bufio.Reader      // 带缓冲的读取器
	closer  io.Closer          // 可选的关闭器，用于关闭底层连接
	format  Format             // 流式响应格式
	parser  ChunkParser        // 块解析器
	ctx     context.Context    // 上下文，用于取消操作
	cancel  context.CancelFunc // 取消函数
	chunks  chan *Chunk        // 块输出通道
	errors  chan error         // 错误通道
	done    chan struct{}      // 完成信号通道
	result  *Result            // 累积的结果
	mu      sync.Mutex         // 保护并发访问
	wg      sync.WaitGroup     // 等待 goroutine 退出
	closed  bool               // 是否已关闭
	started bool               // 是否已启动处理
	onChunk func(*Chunk)       // 块处理回调
	onDone  func(*Result)      // 完成回调
	onError func(error)        // 错误回调
}

// ChunkParser 定义块解析器接口
// 不同的 AI 厂商使用不同的数据格式，需要实现对应的解析器
type ChunkParser interface {
	// Parse 解析原始数据为 Chunk
	// data 是去除 SSE 前缀后的纯数据部分
	// 返回解析后的 Chunk，如果数据无效可返回 nil
	Parse(data []byte) (*Chunk, error)

	// IsDone 判断是否为流结束标记
	// 例如 OpenAI 格式的 "[DONE]" 标记
	IsDone(data []byte) bool
}

// NewStream 创建流式响应处理器
//
// 参数：
//   - r: 数据源，通常是 HTTP 响应的 Body
//   - format: 流式响应格式，决定使用哪种解析器
//
// 返回创建的 Stream 实例，需要调用 Start() 或 Chunks() 开始处理
func NewStream(r io.Reader, format Format) *Stream {
	ctx, cancel := context.WithCancel(context.Background())

	var closer io.Closer
	if c, ok := r.(io.Closer); ok {
		closer = c
	}

	s := &Stream{
		reader: bufio.NewReader(r),
		closer: closer,
		format: format,
		ctx:    ctx,
		cancel: cancel,
		chunks: make(chan *Chunk, 100),
		errors: make(chan error, 1),
		done:   make(chan struct{}),
		result: &Result{},
	}

	// 设置解析器
	switch format {
	case OpenAIFormat:
		s.parser = &OpenAIParser{}
	case ClaudeFormat:
		s.parser = &ClaudeParser{}
	case GeminiFormat:
		s.parser = &GeminiParser{}
	default:
		s.parser = &OpenAIParser{}
	}

	return s
}

// NewStreamWithContext 创建带上下文的流式响应处理器
// 当上下文取消时，流处理会自动停止
//
// 参数：
//   - ctx: 控制流处理生命周期的上下文
//   - r: 数据源
//   - format: 流式响应格式
func NewStreamWithContext(ctx context.Context, r io.Reader, format Format) *Stream {
	s := NewStream(r, format)
	// 释放 NewStream 中创建的默认 cancel，避免 context 泄漏
	s.cancel()
	s.ctx, s.cancel = context.WithCancel(ctx)
	return s
}

// NewStreamWithParser 创建使用自定义解析器的流式响应处理器
// 用于处理非标准格式或自定义协议的流式响应
//
// 参数：
//   - r: 数据源
//   - parser: 自定义的块解析器实现
func NewStreamWithParser(r io.Reader, parser ChunkParser) *Stream {
	s := NewStream(r, CustomFormat)
	s.parser = parser
	return s
}

// SetParser 设置自定义解析器
// 可以在创建 Stream 后替换默认解析器
// 支持链式调用
func (s *Stream) SetParser(parser ChunkParser) *Stream {
	s.parser = parser
	return s
}

// OnChunk 设置块处理回调函数
// 每收到一个有效块时调用此回调
// 适用于需要实时处理每个块的场景，如流式输出到终端
// 必须在 Start() 或 Chunks() 之前调用
// 支持链式调用
func (s *Stream) OnChunk(fn func(*Chunk)) *Stream {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onChunk = fn
	return s
}

// OnDone 设置流处理完成回调函数
// 当流正常结束或遇到结束标记时调用
// 回调参数包含完整的聚合结果
// 必须在 Start() 或 Chunks() 之前调用
// 支持链式调用
func (s *Stream) OnDone(fn func(*Result)) *Stream {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onDone = fn
	return s
}

// OnError 设置错误处理回调函数
// 当解析出错时调用，但不会中断流处理
// 必须在 Start() 或 Chunks() 之前调用
// 支持链式调用
func (s *Stream) OnError(fn func(error)) *Stream {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onError = fn
	return s
}

// Start 开始处理流
// 启动后台 goroutine 读取和解析数据
// 此方法是非阻塞的，立即返回
// 多次调用是安全的，只有首次调用会启动处理
func (s *Stream) Start() *Stream {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return s
	}
	s.started = true
	// 快照回调引用，避免 processLoop 中需要加锁读取
	onChunk := s.onChunk
	onDone := s.onDone
	onError := s.onError
	s.wg.Add(1)
	s.mu.Unlock()

	go func() {
		defer s.wg.Done()
		s.processLoop(onChunk, onDone, onError)
	}()
	return s
}

// Chunks 返回块接收通道
// 自动启动流处理（如果尚未启动）
// 通道在流结束后关闭，可以安全地用 range 遍历
//
// 示例：
//
//	for chunk := range stream.Chunks() {
//	    fmt.Print(chunk.Content)
//	}
func (s *Stream) Chunks() <-chan *Chunk {
	s.Start()
	return s.chunks
}

// Errors 返回错误接收通道
// 用于接收解析过程中的非致命错误
// 通道有缓冲，最多保存一个错误
func (s *Stream) Errors() <-chan error {
	return s.errors
}

// Done 返回完成信号通道
// 当流处理结束（无论成功或失败）时，通道会关闭
// 可用于等待流处理完成
func (s *Stream) Done() <-chan struct{} {
	return s.done
}

// Result 阻塞等待并返回完整结果
// 等待流处理完成后返回聚合的 Result
// 注意：必须先调用 Start() 或 Chunks() 启动处理
func (s *Stream) Result() *Result {
	<-s.done
	s.mu.Lock()
	result := s.result
	s.mu.Unlock()
	return result
}

// Collect 收集完整响应
// 这是最常用的阻塞式 API，自动启动处理并等待完成
// 返回聚合后的完整结果和可能的错误
//
// 示例：
//
//	result, err := stream.Collect()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(result.Content)
func (s *Stream) Collect() (*Result, error) {
	s.Start()

	var lastErr error

	// 等待处理完成，processLoop 已经更新了 s.result
	for {
		select {
		case _, ok := <-s.chunks:
			if !ok {
				// 通道关闭，处理完成
				select {
				case err := <-s.errors:
					if err != nil {
						lastErr = err
					}
				default:
				}
				s.mu.Lock()
				result := s.result
				s.mu.Unlock()
				return result, lastErr
			}
			// chunk 已在 processLoop 中处理，这里只需消费

		case err := <-s.errors:
			lastErr = err

		case <-s.ctx.Done():
			s.mu.Lock()
			result := s.result
			s.mu.Unlock()
			return result, s.ctx.Err()
		}
	}
}

// Close 关闭流并释放资源
// 取消上下文，停止后台处理，等待 goroutine 退出
// 如果底层 Reader 实现了 io.Closer，也会一并关闭
// 多次调用是安全的
func (s *Stream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.cancel()
	closer := s.closer
	s.mu.Unlock()

	var closeErr error
	if closer != nil {
		closeErr = closer.Close()
	}

	// 关闭底层 reader 后再等待，确保阻塞中的 ReadString 能被唤醒。
	s.wg.Wait()
	return closeErr
}

// processLoop 是后台处理的主循环
// 持续从 reader 读取行，解析为 Chunk，发送到通道
// 处理 SSE 格式的 "data:" 前缀
//
// 注意：本循环按「行」分帧——每个 "data:" 行独立解析为一个 chunk，不依赖 SSE
// 空行事件边界。这与 toolkit/net/sse.Reader 的「空行分帧 + 多 data 行 \n 合并」
// 语义不等价：部分上游/网关在相邻 chunk 间不插入空行（见 stream_bridge_test.go
// 的 claudeSSEWithToolUse 夹具），sse.Reader 会把连续 data 行合并成单事件，导致
// 多段 JSON 拼接后 Unmarshal 失败。因此即便 W10 已提供 sse.WithDoneFunc/
// ReadUntilDone 解决了 Provider done 检测，事件分帧粒度差异仍未被覆盖，本路径
// 保留本地行级实现（正确性优先，详见 sse_boundary_test.go 头注与该夹具）。
func (s *Stream) processLoop(onChunk func(*Chunk), onDone func(*Result), onError func(error)) {
	defer close(s.chunks)
	defer close(s.done)

	var contentBuf bytes.Buffer

	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}

		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				s.sendErrorWithCallback(err, onError)
			}
			s.finalize(contentBuf.String(), onDone)
			return
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// 处理 SSE 格式
		if data, found := strings.CutPrefix(line, "data:"); found {
			data = strings.TrimSpace(data)

			// 先解析数据，再判断是否结束
			// 这样可以确保最后一个包含内容的 chunk 不会被丢弃
			// （Gemini 的最后一个 chunk 既包含 content 又包含 finishReason）
			chunk, err := s.parser.Parse([]byte(data))
			if err != nil {
				// 如果解析失败且是结束标记，则正常结束
				if s.parser.IsDone([]byte(data)) {
					s.finalize(contentBuf.String(), onDone)
					return
				}
				s.sendErrorWithCallback(err, onError)
				continue
			}

			if chunk != nil {
				contentBuf.WriteString(chunk.Content)

				// 更新结果（加锁保护）
				s.mu.Lock()
				s.result.Chunks = append(s.result.Chunks, chunk)
				if chunk.ID != "" && s.result.ID == "" {
					s.result.ID = chunk.ID
				}
				if chunk.Role != "" && s.result.Role == "" {
					s.result.Role = chunk.Role
				}
				if chunk.Model != "" && s.result.Model == "" {
					s.result.Model = chunk.Model
				}
				if chunk.FinishReason != "" {
					s.result.FinishReason = chunk.FinishReason
				}
				if len(chunk.ToolCalls) > 0 {
					s.result.ToolCalls = mergeToolCalls(s.result.ToolCalls, chunk.ToolCalls)
				}
				s.mu.Unlock()

				// 回调
				if onChunk != nil {
					onChunk(chunk)
				}

				// 发送到通道
				select {
				case s.chunks <- chunk:
				case <-s.ctx.Done():
					return
				}

				// 在发送 chunk 后检查是否结束
				// 这确保了最后一个有内容的 chunk 被正确处理
				if s.parser.IsDone([]byte(data)) {
					s.finalize(contentBuf.String(), onDone)
					return
				}
			}
		}
	}
}

// finalize 收尾流处理：把累计正文落入 result，做工具调用兜底还原，再触发 onDone。
//
// processLoop 的三处结束分支（EOF/读错、解析失败遇 DONE、正常遇 DONE）共用本方法，
// 保证收尾逻辑单点一致。兜底还原：部分模型（DeepSeek 经 OpenAI 兼容网关）把工具调用以
// 文本标记内嵌进 content 而非结构化 tool_calls 字段，流式路径同样会把用户意图当普通文本
// 丢弃——这里在结构化字段为空时还原结构化 ToolCalls 并剥净正文。见 inline_toolcalls.go。
func (s *Stream) finalize(content string, onDone func(*Result)) {
	s.mu.Lock()
	s.result.Content = content
	if len(s.result.ToolCalls) == 0 {
		if calls, cleaned, ok := RecoverInlineToolCalls(s.result.Content); ok {
			s.result.ToolCalls = calls
			s.result.Content = cleaned
			if s.result.FinishReason == "" || s.result.FinishReason == "stop" {
				s.result.FinishReason = "tool_calls"
			}
		}
	}
	result := s.result
	s.mu.Unlock()
	if onDone != nil {
		onDone(result)
	}
}

// sendErrorWithCallback 发送错误到错误通道并触发回调
// 错误通道有缓冲但不阻塞，如果通道满则丢弃
func (s *Stream) sendErrorWithCallback(err error, onError func(error)) {
	if onError != nil {
		onError(err)
	}
	select {
	case s.errors <- err:
	default:
	}
}

// mergeToolCalls 合并工具调用列表
//
// OpenAI streaming 协议规定（也被 OpenRouter / new-api / one-api 等兼容网关沿用）：
//   - 首个 tool_call delta 携带 index + id + type + function.name + function.arguments(可为空)
//   - 后续 delta 仅携带 index + function.arguments 增量片段，id/name/type 均为空
//   - 多个并行 tool_call 用不同的 index 区分
//
// 因此**合并键必须是 Index 而不是 ID**。按 ID 合并的旧实现遇到 Claude / DeepSeek-R1 /
// Qwen3 这类按 token 切片 streaming 的模型必出"空 name 工具调用"，畸形请求回灌
// 二轮 prompt 被网关拒收（参考 BUG-20260520）。
func mergeToolCalls(existing, new []ToolCall) []ToolCall {
	if len(new) == 0 {
		return existing
	}

	for _, tc := range new {
		found := false
		for i := range existing {
			if existing[i].Index == tc.Index {
				existing[i].Arguments += tc.Arguments
				if tc.ID != "" {
					existing[i].ID = tc.ID
				}
				if tc.Name != "" {
					existing[i].Name = tc.Name
				}
				if tc.Type != "" {
					existing[i].Type = tc.Type
				}
				found = true
				break
			}
		}
		if !found {
			existing = append(existing, tc)
		}
	}
	return existing
}

// ============== 便捷函数 ==============

// CollectContent 收集流式响应的完整内容
// 这是最简单的使用方式，只关心最终的文本内容
//
// 参数：
//   - r: 数据源
//   - format: 流式响应格式
//
// 返回拼接后的完整文本内容
func CollectContent(r io.Reader, format Format) (string, error) {
	stream := NewStream(r, format)
	result, err := stream.Collect()
	if err != nil {
		return "", err
	}
	return result.Content, nil
}

// ProcessStream 使用自定义处理函数处理流式响应
// 逐块调用 handler 函数，如果 handler 返回错误则停止处理
//
// 参数：
//   - r: 数据源
//   - format: 流式响应格式
//   - handler: 块处理函数，返回非 nil 错误时停止处理
//
// 返回 handler 返回的错误或流处理过程中的错误
func ProcessStream(r io.Reader, format Format, handler func(*Chunk) error) error {
	stream := NewStream(r, format)

	for chunk := range stream.Chunks() {
		if err := handler(chunk); err != nil {
			stream.Close()
			return err
		}
	}

	select {
	case err := <-stream.Errors():
		return err
	default:
		return nil
	}
}
