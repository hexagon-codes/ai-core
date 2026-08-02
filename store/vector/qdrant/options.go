package qdrant

import "time"

// Option Qdrant 存储选项
type Option func(*Config)

// WithHost 设置服务器地址
func WithHost(host string) Option {
	return func(c *Config) {
		c.Host = host
	}
}

// WithPort 设置服务器端口
func WithPort(port int) Option {
	return func(c *Config) {
		c.Port = port
	}
}

// WithCollection 设置集合名称
func WithCollection(name string) Option {
	return func(c *Config) {
		c.Collection = name
	}
}

// WithDimension 设置向量维度
func WithDimension(dim int) Option {
	return func(c *Config) {
		c.Dimension = dim
	}
}

// WithAPIKey 设置 API 密钥
func WithAPIKey(key string) Option {
	return func(c *Config) {
		c.APIKey = key
	}
}

// WithHTTPS 设置是否使用 HTTPS
func WithHTTPS(https bool) Option {
	return func(c *Config) {
		c.HTTPS = https
	}
}

// WithTimeout 设置请求超时时间
func WithTimeout(timeout time.Duration) Option {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

// WithDistance 设置距离度量方式
func WithDistance(distance Distance) Option {
	return func(c *Config) {
		c.Distance = distance
	}
}

// WithOnDisk 设置是否将向量存储在磁盘上
func WithOnDisk(onDisk bool) Option {
	return func(c *Config) {
		c.OnDisk = onDisk
	}
}

// WithCreateCollection 设置是否自动创建集合
func WithCreateCollection(create bool) Option {
	return func(c *Config) {
		c.CreateCollection = create
	}
}

// WithPointIDStrategy selects the persisted Qdrant point-ID encoding.
func WithPointIDStrategy(strategy PointIDStrategy) Option {
	return func(c *Config) {
		c.PointIDStrategy = strategy
	}
}

// WithMaxResponseBytes sets the maximum accepted Qdrant HTTP response body.
func WithMaxResponseBytes(limit int64) Option {
	return func(c *Config) {
		c.MaxResponseBytes = limit
	}
}

// NewWithOptions 使用选项创建 Qdrant 存储
//
// 示例：
//
//	store, err := qdrant.NewWithOptions(
//	    qdrant.WithHost("localhost"),
//	    qdrant.WithPort(6333),
//	    qdrant.WithCollection("documents"),
//	    qdrant.WithDimension(1536),
//	    qdrant.WithCreateCollection(true),
//	)
func NewWithOptions(opts ...Option) (*Store, error) {
	cfg := Config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return New(cfg)
}
