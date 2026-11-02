package llm

import (
	"encoding/json"
	"strings"
)

// receiptScanner 逐字节跳过正文，只保留有界字段名/标量和有限嵌套路径。
// 观察上限仅令该字段保持未知，不影响原响应字节、解析器或正常业务。
type receiptScanner struct {
	stack        []receiptContainer
	token        []byte
	stringToken  bool
	escaped      bool
	truncated    bool
	primitive    bool
	skipDepth    int
	rootComplete bool
	fact         func([]string, string, bool)
}
type receiptContainer struct {
	kind  byte
	path  []string
	key   string
	state byte
}

func newReceiptScanner(fact func([]string, string, bool)) *receiptScanner {
	return &receiptScanner{fact: fact}
}
func (s *receiptScanner) path() []string {
	if len(s.stack) == 0 {
		return nil
	}
	top := s.stack[len(s.stack)-1]
	path := append([]string(nil), top.path...)
	if top.kind == '{' {
		path = append(path, top.key)
	}
	return path
}
func (s *receiptScanner) completeValue() {
	if len(s.stack) > 0 {
		s.stack[len(s.stack)-1].state = 'c'
	}
}
func (s *receiptScanner) scalar(value string, numeric bool) {
	if len(s.stack) == 0 {
		return
	}
	top := &s.stack[len(s.stack)-1]
	if top.kind == '{' && top.state == 'k' {
		if !numeric {
			top.key = value
			top.state = ':'
		}
		return
	}
	if top.state == 'v' {
		s.fact(s.path(), value, numeric)
		s.completeValue()
	}
}
func (s *receiptScanner) Feed(data []byte) {
	for _, character := range data {
		if s.stringToken {
			if len(s.token) < 4096 {
				s.token = append(s.token, character)
			} else {
				s.truncated = true
			}
			if s.escaped {
				s.escaped = false
				continue
			}
			if character == '\\' {
				s.escaped = true
				continue
			}
			if character != '"' {
				continue
			}
			s.stringToken = false
			if !s.truncated && s.skipDepth == 0 {
				var value string
				if json.Unmarshal(s.token, &value) == nil {
					s.scalar(value, false)
				}
			} else if s.skipDepth == 0 {
				s.completeValue()
			}
			s.token = s.token[:0]
			s.truncated = false
			continue
		}
		if s.primitive {
			if character != ',' && character != '}' && character != ']' && character != ' ' && character != '\n' && character != '\r' && character != '\t' {
				if len(s.token) < 128 {
					s.token = append(s.token, character)
				} else {
					s.truncated = true
				}
				continue
			}
			if !s.truncated && s.skipDepth == 0 {
				s.scalar(string(s.token), true)
			}
			s.primitive = false
			s.token = s.token[:0]
			s.truncated = false
		}
		switch character {
		case '"':
			s.stringToken = true
			s.token = append(s.token[:0], '"')
			s.escaped = false
		case '{', '[':
			if len(s.stack) == 0 {
				s.rootComplete = false
			}
			if s.skipDepth > 0 {
				s.skipDepth++
				continue
			}
			if len(s.stack) >= 128 {
				s.skipDepth = 1
				continue
			}
			path := s.path()
			state := byte('v')
			if character == '{' {
				state = 'k'
			}
			s.stack = append(s.stack, receiptContainer{kind: character, path: path, state: state})
		case '}', ']':
			if s.skipDepth > 0 {
				s.skipDepth--
				if s.skipDepth == 0 {
					s.completeValue()
				}
				continue
			}
			if len(s.stack) > 0 {
				s.stack = s.stack[:len(s.stack)-1]
				s.completeValue()
				if len(s.stack) == 0 {
					s.rootComplete = true
				}
			}
		case ':':
			if len(s.stack) > 0 && s.stack[len(s.stack)-1].state == ':' {
				s.stack[len(s.stack)-1].state = 'v'
			}
		case ',':
			if len(s.stack) > 0 {
				top := &s.stack[len(s.stack)-1]
				if top.kind == '{' {
					top.state = 'k'
				} else {
					top.state = 'v'
				}
			}
		default:
			if s.skipDepth == 0 && len(s.stack) > 0 && strings.ContainsRune("-0123456789tfn", rune(character)) {
				s.primitive = true
				s.token = append(s.token[:0], character)
			}
		}
	}
}
