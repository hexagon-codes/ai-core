package catalog_test

import (
	"testing"

	"github.com/hexagon-codes/ai-core/catalog"
	llmcompat "github.com/hexagon-codes/ai-core/llm/compatible"
	imagegen "github.com/hexagon-codes/ai-core/media/image"
	videogen "github.com/hexagon-codes/ai-core/media/video"
	"github.com/hexagon-codes/ai-core/media/voice"
)

type capabilityProvider interface {
	Name() string
	Capabilities() []catalog.Capability
}

func TestNewProvidersExposeCapabilities(t *testing.T) {
	doubaoCN, err := llmcompat.NewProfile(llmcompat.ProfileDoubaoCN, "key")
	if err != nil {
		t.Fatal(err)
	}
	doubaoGlobal, err := llmcompat.NewProfile(llmcompat.ProfileDoubaoGlobal, "key")
	if err != nil {
		t.Fatal(err)
	}

	providers := []capabilityProvider{
		doubaoCN,
		doubaoGlobal,
		videogen.NewSeedanceCN("key").(capabilityProvider),
		videogen.NewSeedanceGlobal("key").(capabilityProvider),
		videogen.NewKling("token", "").(capabilityProvider),
		videogen.NewVidu("key").(capabilityProvider),
		videogen.NewVeo("key").(capabilityProvider),
		videogen.NewMidjourneyCompatible("key", "https://example.com").(capabilityProvider),
		imagegen.NewFlux("key").(capabilityProvider),
		imagegen.NewMidjourneyCompatible("key", "https://example.com").(capabilityProvider),
		voice.NewElevenLabsTTS("key"),
	}

	for _, p := range providers {
		t.Run(p.Name(), func(t *testing.T) {
			rows := p.Capabilities()
			if len(rows) == 0 {
				t.Fatal("expected capabilities")
			}
			for _, row := range rows {
				if row.Provider == "" || row.Model == "" || row.Modality == "" {
					t.Fatalf("incomplete capability row: %+v", row)
				}
			}
		})
	}
}

func TestRegionalDoubaoCapabilitiesAreDistinct(t *testing.T) {
	cn, err := llmcompat.NewProfile(llmcompat.ProfileDoubaoCN, "key")
	if err != nil {
		t.Fatal(err)
	}
	global, err := llmcompat.NewProfile(llmcompat.ProfileDoubaoGlobal, "key")
	if err != nil {
		t.Fatal(err)
	}

	if !hasFeature(cn.Capabilities(), "region:cn") {
		t.Fatalf("doubao-cn missing region feature: %+v", cn.Capabilities())
	}
	if !hasFeature(global.Capabilities(), "region:global") {
		t.Fatalf("doubao-global missing region feature: %+v", global.Capabilities())
	}
}

func hasFeature(rows []catalog.Capability, feature string) bool {
	for _, row := range rows {
		for _, got := range row.Features {
			if got == feature {
				return true
			}
		}
	}
	return false
}
