package streamx

// Reader 将通道式 Stream 适配为泛型 StreamReader[*Chunk]。
//
// Stream 自身只提供「通道 / 回调 / Collect」三种消费方式，不支持 Copy/Map
// 等组合算子。通过本桥接，调用方可以复用 StreamReader 的 Copy/Map/Filter/Recv
// 能力——典型场景是在不破坏原始流的前提下做工具调用预检（见 Copy + Checker）。
//
// 语义：
//   - 返回后由 reader 接管 chunk 消费；内部启动一个 goroutine 把 Chunks()
//     转发到 Pipe；
//   - 上游流正常结束时关闭 reader（Recv 返回 io.EOF）；
//   - 上游有非致命错误时经 CloseWithError 传播给 reader；
//   - 若下游 reader 已关闭（Send 失败），则关闭上游 Stream 以停止处理，避免泄漏。
//
// 典型用法（工具调用检测，保留原始流给下游）：
//
//	sr := provider.Stream(ctx, req).Reader()
//	cp := sr.Copy(2)
//	has, _ := llm.DefaultStreamToolCallChecker(ctx, cp[0])
//	// cp[1] 仍是完整流，交给下游消费
func (s *Stream) Reader() *StreamReader[*Chunk] {
	r, w := Pipe[*Chunk](16)
	go func() {
		// Chunks() 会启动处理，并在流结束时关闭通道
		for c := range s.Chunks() {
			if err := w.Send(c); err != nil {
				// 下游已关闭，停止上游处理避免 goroutine 泄漏
				_ = s.Close()
				return
			}
		}
		// chunk 通道关闭后，若有缓冲的非致命错误则传播
		select {
		case err := <-s.Errors():
			if err != nil {
				_ = w.CloseWithError(err)
				return
			}
		default:
		}
		_ = w.Close()
	}()
	return r
}
