package streamx

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

// TestGRPCFrame_RoundTrip 编码两条消息后逐帧读回，载荷与顺序一致。
func TestGRPCFrame_RoundTrip(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(EncodeGRPCFrame([]byte("hello")))
	buf.Write(EncodeGRPCFrame([]byte("world")))

	fr := NewGRPCFrameReader(&buf)

	m1, err := fr.Next()
	if err != nil || string(m1) != "hello" {
		t.Fatalf("帧1 = (%q,%v), want (hello,nil)", m1, err)
	}
	m2, err := fr.Next()
	if err != nil || string(m2) != "world" {
		t.Fatalf("帧2 = (%q,%v), want (world,nil)", m2, err)
	}
	if _, err := fr.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("帧边界结束应返回 io.EOF, got %v", err)
	}
}

// TestGRPCFrame_Empty 空载荷帧合法（长度 0）。
func TestGRPCFrame_Empty(t *testing.T) {
	buf := bytes.NewReader(EncodeGRPCFrame(nil))
	fr := NewGRPCFrameReader(buf)
	m, err := fr.Next()
	if err != nil || len(m) != 0 {
		t.Errorf("空帧 = (%v,%v), want (空,nil)", m, err)
	}
}

// TestGRPCFrame_Compressed 压缩标志为 1 时返回 ErrGRPCCompressed。
func TestGRPCFrame_Compressed(t *testing.T) {
	frame := EncodeGRPCFrame([]byte("x"))
	frame[0] = 1 // 标记为压缩
	fr := NewGRPCFrameReader(bytes.NewReader(frame))
	if _, err := fr.Next(); !errors.Is(err, ErrGRPCCompressed) {
		t.Errorf("压缩帧应返回 ErrGRPCCompressed, got %v", err)
	}
}

// TestGRPCFrame_Truncated 消息体被截断时返回 ErrUnexpectedEOF。
func TestGRPCFrame_Truncated(t *testing.T) {
	full := EncodeGRPCFrame([]byte("hello"))
	truncated := full[:len(full)-2] // 砍掉尾部 2 字节
	fr := NewGRPCFrameReader(bytes.NewReader(truncated))
	if _, err := fr.Next(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("截断帧应返回 ErrUnexpectedEOF, got %v", err)
	}
}

// TestReadGRPCFrames_Stream 流式读取多帧到 StreamReader，顺序正确并以 EOF 收尾。
func TestReadGRPCFrames_Stream(t *testing.T) {
	var buf bytes.Buffer
	for _, s := range []string{"a", "bb", "ccc"} {
		buf.Write(EncodeGRPCFrame([]byte(s)))
	}

	sr := ReadGRPCFrames(&buf)
	var got []string
	for {
		m, err := sr.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, string(m))
	}
	if len(got) != 3 || got[0] != "a" || got[1] != "bb" || got[2] != "ccc" {
		t.Errorf("流式帧 = %v, want [a bb ccc]", got)
	}
}

// TestReadGRPCFrames_ErrorPropagation 截断流的错误经 StreamReader 传播。
func TestReadGRPCFrames_ErrorPropagation(t *testing.T) {
	full := EncodeGRPCFrame([]byte("hello"))
	sr := ReadGRPCFrames(bytes.NewReader(full[:len(full)-2]))
	_, err := sr.Recv()
	if err == nil || errors.Is(err, io.EOF) {
		t.Errorf("截断流应传播非 EOF 错误, got %v", err)
	}
}
