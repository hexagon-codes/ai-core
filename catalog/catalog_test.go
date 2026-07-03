package catalog

import "testing"

func TestRegistryFindsByModalityAndFeature(t *testing.T) {
	r := NewRegistry(
		Capability{
			Provider: "openai",
			Model:    "gpt-4o",
			Modality: ModalityText,
			Features: []string{FeatureStreaming, FeatureVision, FeatureStreaming},
		},
		Capability{
			Provider: "bfl",
			Model:    "flux-pro-1.1",
			Modality: ModalityImage,
			Features: []string{FeatureAsyncTask},
		},
	)

	rows := r.Find(Query{Modality: ModalityText, Feature: FeatureVision})
	if len(rows) != 1 || rows[0].Provider != "openai" {
		t.Fatalf("unexpected rows: %+v", rows)
	}
	if len(rows[0].Features) != 2 {
		t.Fatalf("features should be normalized/deduplicated, got %+v", rows[0].Features)
	}
}

func TestRegistryRegisterProvider(t *testing.T) {
	p := fakeProvider{}
	var r Registry
	r.RegisterProvider(p)

	rows := r.All()
	if len(rows) != 1 || rows[0].Provider != "vidu" {
		t.Fatalf("unexpected rows: %+v", rows)
	}
}

type fakeProvider struct{}

func (fakeProvider) Capabilities() []Capability {
	return []Capability{{
		Provider: "vidu",
		Model:    "vidu2.0",
		Modality: ModalityVideo,
	}}
}
