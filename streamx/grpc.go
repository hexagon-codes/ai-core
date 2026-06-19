package streamx

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
)

// gRPC 解析层：解析 gRPC 的"长度前缀消息"（Length-Prefixed-Message）线格式。
//
// gRPC 在 HTTP/2 DATA 帧中承载的消息线格式为（见 gRPC over HTTP/2 规范）：
//
//	Length-Prefixed-Message → Compressed-Flag(1B) Message-Length(4B 大端) Message
//
// 与 SSE（文本、按 data: 行解析）不同，gRPC 是二进制、按长度前缀分帧。本层只做
// **分帧（framing）**：把字节流切分为一条条消息体；消息体的 protobuf 解码由调用方
// 按自身的消息类型完成（不同上游 proto 定义不同，本层保持 AI/proto 无关）。

// grpcHeaderLen 是 gRPC 长度前缀帧头长度（1 字节压缩标志 + 4 字节长度）。
const grpcHeaderLen = 5

// ErrGRPCCompressed 表示遇到压缩帧（压缩标志为 1）。
//
// 本层不内置解压（解压算法由 grpc-encoding 头协商，属传输层职责）；调用方需在
// 上层先解压，或确保对端不压缩。返回明确错误而非静默吐出压缩字节。
var ErrGRPCCompressed = errors.New("streamx: gRPC 压缩帧需调用方先解压")

// GRPCFrameReader 从底层字节流逐帧读取 gRPC 长度前缀消息。
type GRPCFrameReader struct {
	r *bufio.Reader
}

// NewGRPCFrameReader 创建 gRPC 帧读取器。
func NewGRPCFrameReader(r io.Reader) *GRPCFrameReader {
	return &GRPCFrameReader{r: bufio.NewReader(r)}
}

// Next 读取下一帧，返回消息体（已剥离 5 字节帧头）。
//
//   - 流在帧边界正常结束 → 返回 io.EOF；
//   - 帧头/消息体被截断 → 返回 io.ErrUnexpectedEOF；
//   - 压缩帧 → 返回 ErrGRPCCompressed。
func (g *GRPCFrameReader) Next() ([]byte, error) {
	var header [grpcHeaderLen]byte
	if _, err := io.ReadFull(g.r, header[:]); err != nil {
		// 帧边界处无更多数据时 io.ReadFull 返回 io.EOF；部分读取则为 ErrUnexpectedEOF。
		return nil, err
	}

	if header[0] != 0 {
		return nil, ErrGRPCCompressed
	}

	length := binary.BigEndian.Uint32(header[1:grpcHeaderLen])
	if length == 0 {
		return []byte{}, nil
	}

	msg := make([]byte, length)
	if _, err := io.ReadFull(g.r, msg); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return msg, nil
}

// EncodeGRPCFrame 把消息体编码为 gRPC 未压缩长度前缀帧（5 字节头 + 消息体）。
//
// 主要用于测试与对称编码；生产中发送方框架通常已处理分帧。
func EncodeGRPCFrame(payload []byte) []byte {
	frame := make([]byte, grpcHeaderLen+len(payload))
	frame[0] = 0 // 未压缩
	binary.BigEndian.PutUint32(frame[1:grpcHeaderLen], uint32(len(payload)))
	copy(frame[grpcHeaderLen:], payload)
	return frame
}

// ReadGRPCFrames 从 r 持续读取 gRPC 帧，把每帧消息体送入 StreamReader[[]byte]，
// 从而复用 StreamReader 的 Recv/Map/Filter 等组合算子。
//
// 读到 io.EOF 正常结束（关闭流）；其它错误经 CloseWithError 传播给消费者。
func ReadGRPCFrames(r io.Reader) *StreamReader[[]byte] {
	reader, writer := Pipe[[]byte](16)
	fr := NewGRPCFrameReader(r)

	go func() {
		for {
			msg, err := fr.Next()
			if err != nil {
				if errors.Is(err, io.EOF) {
					_ = writer.Close()
				} else {
					_ = writer.CloseWithError(err)
				}
				return
			}
			if writer.Send(msg) != nil {
				return // 下游已关闭
			}
		}
	}()

	return reader
}
