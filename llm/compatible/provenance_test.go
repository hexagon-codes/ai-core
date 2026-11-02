package compatible

import "testing"

func TestConfiguredEndpointKeepsDeclarationProvenance(t *testing.T) {
	p, err := NewProfile(ProfileDoubaoGlobal, "unused", WithBaseURL("http://127.0.0.1:12345"))
	if err != nil {
		t.Fatal(err)
	}
	rows := p.Capabilities()
	if len(rows) == 0 {
		t.Fatal("profile declarations disappeared")
	}
	for _, row := range rows {
		if row.OfficialAPI || row.Stable || row.Source == "" || row.Evidence != "declared" || row.EndpointSource != "configured" {
			t.Fatalf("configured endpoint provenance = %+v", row)
		}
	}
}

func TestConfiguredModelsDoNotInheritOpenAICatalog(t *testing.T) {
	if models := New("custom", "unused").Models(); len(models) != 0 {
		t.Fatalf("unconfigured models = %+v", models)
	}
	p := New("custom", "unused", WithModel("sample-model"))
	models := p.Models()
	if len(models) != 1 || models[0].ID != "sample-model" || len(models[0].Features) != 0 {
		t.Fatalf("configured models = %+v", models)
	}
	rows := p.Capabilities()
	if len(rows) != 1 || rows[0].OfficialAPI || rows[0].Stable || rows[0].Evidence != "declared" {
		t.Fatalf("configured capability = %+v", rows)
	}
}
