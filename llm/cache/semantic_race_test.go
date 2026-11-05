package cache

import (
	"sync"
	"testing"
	"time"
)

func TestSemanticCacheConcurrentReconfigure(t *testing.T) {
	t.Parallel()

	cache := NewSemanticCache(SemanticOptions{Enabled: true, TTL: time.Hour, MaxEntries: 32})

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				cache.Put("input", "response", "provider", "model")
				cache.Get("input", "provider", "model")
				_, _ = cache.Do("input", "provider", "model", func() (string, string, error) {
					return "response", "model", nil
				})
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 200; j++ {
			cache.Reconfigure(SemanticOptions{Enabled: j%2 == 0, TTL: time.Hour, MaxEntries: 32})
		}
	}()
	wg.Wait()
}
